import gym
import torch
from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import cv2
import numpy as np

# Hyperparameters
n_train_processes = 6
learning_rate = 8e-5
update_interval = 20
gamma = 0.99
lam = 1.0
max_train_ep = int(15000/n_train_processes) + 1


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7744, 512)
        self.fc_pi = nn.Linear(512, 6)
        self.fc_v = nn.Linear(512, 1)

    def pi(self, x, softmax_dim=-1):
        # x = x/255.0
        x = F.relu(self.conv1(x))       #使用relu函数会导致某些step，v值输出过大或过小
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 7744)))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        # print(prob)
        return prob

    def v(self, x):
        # x = x/255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 7744)))
        v = self.fc_v(x)
        return v


def ColorMat2Binary(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return state

    # state_resize = cv2.resize(state, (80, 80))
    # state_gray = cv2.cvtColor(state_resize, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    # _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)  # 转换为二值图
    # return state_binary


def train(global_model, rank, global_ep, global_r_lst):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3000, gamma=0.6)

    env = gym.make('PongDeterministic-v4')
    env.seed(1)
    max_step = 10000

    for n_epi in range(max_train_ep):
        step = 0
        ep_r = 0
        done = False
        s = ColorMat2Binary(env.reset())
        s_shadow = np.stack((s, s, s, s), axis=0)
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                # if rank == 0:
                #     env.render()
                prob = local_model.pi(torch.unsqueeze(torch.FloatTensor(s_shadow), 0))
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                s_prime = np.reshape(ColorMat2Binary(s_prime), (1, 84, 84))

                s_prime_shadow = np.append(s_shadow[1:, :, :], s_prime, axis=0)
                ep_r += r
                step += 1
                s_lst.append(s_shadow)
                a_lst.append([a])
                r_lst.append([r])

                s_shadow = s_prime_shadow
                if done or step >= max_step:
                    break
            s_final = torch.unsqueeze(torch.FloatTensor(s_prime_shadow), 0)
            R = 0.0 if done else local_model.v(s_final).item()
            s_batch, a_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst)
            values = local_model.v(s_batch).detach().numpy()
            r_lst = np.clip(r_lst, -1, 1)
            tds = r_lst + gamma * np.append(values[1:], [[R]], axis=0) - values
            advantage = signal.lfilter([1.0], [1.0, -gamma * lam], tds[::-1])[::-1]
            td_target = advantage + values
            advantage = torch.tensor(advantage.copy(), dtype=torch.float)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            td_target = torch.tensor(td_target.copy(), dtype=torch.float)

            pi = local_model.pi(s_batch)
            actions_log_probs = torch.log(pi.gather(1, a_batch) + 1e-8)
            policy_loss = -((actions_log_probs * advantage.detach()).sum())
            value_delta = local_model.v(s_batch) - td_target.detach()
            value_loss = torch.mul(value_delta, value_delta).sum()
            entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).sum()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param.grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

        global_ep.value += 1
        scheduler.step()
        print(rank, n_epi, ep_r, loss.item(), global_ep.value)
        global_r_lst.append(ep_r)
        if global_ep.value % 100 == 0:
            np.save('data/global_epr.npy', np.array(global_r_lst))
        if n_epi % 100 == 0 and rank < 6:
            torch.save(local_model, 'model/' + str(rank) + '_AC.pth')
    env.close()
    print("Training process {} reached maximum episode.".format(rank))



if __name__ == '__main__':
    global_model = ActorCritic()
    global_model.share_memory()
    global_ep = mp.Manager().Value('i', 0)
    global_r_lst = mp.Manager().list()

    processes = []
    for rank in range(n_train_processes):
        p = mp.Process(target=train, args=(global_model, rank, global_ep, global_r_lst,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
