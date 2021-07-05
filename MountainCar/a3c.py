import gym
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np

# Hyperparameters
n_train_processes = 1
learning_rate = 5e-4
update_interval = 20
gamma = 0.99
lam = 0.95
max_train_ep = 200


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        hidden = 100
        self.fc1 = nn.Linear(2, hidden)
        self.fc_pi = nn.Linear(hidden, 3)
        self.fc_v = nn.Linear(hidden, 1)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank, global_ep, global_r_lst):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(1)

    for n_epi in range(max_train_ep):
        ep_r = 0
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                # env.render()
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                ep_r += r

                if done:
                    r = 10
                elif s_prime[0] >= 0.4:
                    r = 1

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r / 10.0])

                s = s_prime
                if done:
                    break
            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            s_batch, a_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst)
            values = local_model.v(s_batch).detach().numpy()
            tds = r_lst + gamma * np.append(values[1:], [[R]], axis=0) - values
            advantage = signal.lfilter([1.0], [1.0, -gamma * lam], tds[::-1])[::-1]
            td_target = advantage + values
            advantage = torch.tensor(advantage.copy(), dtype=torch.float)
            td_target = torch.tensor(td_target.copy(), dtype=torch.float)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            policy_distri = Categorical(pi)
            policy_entropy = policy_distri.entropy()
            entropy = policy_entropy.sum()
            loss = -torch.log(pi_a) * advantage.detach() + \
                   0.5 * F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()) - \
                   0.01 * entropy

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        global_ep.value += 1
        print(rank, n_epi, ep_r, global_ep.value)
        global_r_lst.append(ep_r)
    env.close()
    print("Training process {} reached maximum episode.".format(rank))
    torch.save(local_model, 'model/' + str(rank) + '_AC.pth')
    print(len(global_r_lst))
    np.save('data/global_epr.npy', np.array(global_r_lst))


if __name__ == '__main__':
    global_model = ActorCritic()
    global_model.share_memory()
    global_ep = mp.Manager().Value('i', 0)
    global_r_lst = mp.Manager().list()

    processes = []
    for rank in range(n_train_processes):  # + 1 for test process
        p = mp.Process(target=train, args=(global_model, rank, global_ep, global_r_lst,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
