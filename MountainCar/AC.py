import os
import gym
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from a3c import ActorCritic

# Hyperparameters
learning_rate = 5e-4
update_interval = 20
gamma = 0.99
lam = 0.95
max_train_ep = 800


class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        hidden = 23
        self.fc1 = nn.Linear(2, hidden)
        self.fc_pi = nn.Linear(hidden, 3)
        # self.fc_v = nn.Linear(hidden, 1)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # def v(self, x):
    #     x = F.relu(self.fc1(x))
    #     v = self.fc_v(x)
    #     return v


def softmax(x):
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


def load_teacher(env):
    pth_list = os.listdir('model/')
    teachers = []
    for file in pth_list:
        teacher = torch.load('model/' + file).cuda()
        for param in teacher.parameters():  # 冻结参数
            param.requires_grad = False
        teachers.append(teacher)

    rank = []
    for teacher in teachers:
        total_r = 0
        for episode in range(5):
            obs = env.reset()
            max_step = 200
            for _ in range(max_step):
                action = teacher.pi(torch.tensor(obs, device='cuda', dtype=torch.float)).argmax().item()
                obs, reward, done, _ = env.step(action)
                total_r += reward
                if done:
                    break
        rank.append(-1 / total_r)
    rank = softmax(np.array(rank))
    print(rank)
    return rank, teachers


def train(global_model, global_r_lst):
    local_model = AC().cuda()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    alpha = 0.5
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(1)
    rank, teachers = load_teacher(env)

    for n_epi in range(max_train_ep):
        ep_r = 0
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                # env.render()
                prob = local_model.pi(torch.from_numpy(s).float().cuda())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                ep_r += r

                if done:
                    r = 10
                elif s_prime[0] >= 0.3:
                    r = 1
                elif s_prime[0] >= -0.2:
                    r = 0

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r / 10.0])

                s = s_prime
                if done:
                    break
            s_final = torch.tensor(s_prime, dtype=torch.float, device='cuda')

            idx = np.random.choice(len(rank), 1, p=rank)[0]
            teacher = teachers[idx]

            R = 0.0 if done else teacher.v(s_final).item()
            s_batch, a_batch = torch.tensor(s_lst, dtype=torch.float, device='cuda'), torch.tensor(a_lst, device='cuda')
            values = teacher.v(s_batch).cpu().detach().numpy()
            tds = r_lst + gamma * np.append(values[1:], [[R]], axis=0) - values
            advantage = signal.lfilter([1.0], [1.0, -gamma * lam], tds[::-1])[::-1]
            advantage = torch.tensor(advantage.copy(), dtype=torch.float, device='cuda')

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            policy_distri = Categorical(pi)
            policy_entropy = policy_distri.entropy()
            entropy = policy_entropy.sum()
            rl_loss = -torch.log(pi_a) * advantage.detach() - 0.01 * entropy

            teacher_prob = teacher.pi(s_batch, softmax_dim=1)
            KL = torch.nn.KLDivLoss()
            teach_loss = KL(pi, teacher_prob)
            loss = (1-alpha) * rl_loss + alpha * teach_loss

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        print(n_epi, ep_r)
        global_r_lst.append(ep_r)
        if n_epi > 10:
            alpha = 0.1
        elif n_epi > 50:
            alpha = 0.01
        elif n_epi > 100:
            alpha = 0
    env.close()
    print(len(global_r_lst))
    np.save('data/student20.npy', np.array(global_r_lst))


if __name__ == '__main__':
    global_model = AC().cuda()
    global_model.share_memory()
    global global_r_lst
    global_r_lst = []
    train(global_model, global_r_lst)
