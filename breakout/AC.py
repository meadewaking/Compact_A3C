import os
import gym
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2
from A3C import ActorCritic

# Hyperparameters
learning_rate = 1e-4
update_interval = 20
gamma = 0.99
lam = 1.0
max_train_ep = 15005


class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(3872, 465)
        self.fc_pi = nn.Linear(465, 4)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 3872)))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob



def softmax(x):
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


def load_teacher(env):
    pth_list = os.listdir('model/')
    print(pth_list)
    teachers = []
    for file in pth_list:
        if '_AC' in file:
            teacher = torch.load('model/' + file).cuda()
            for param in teacher.parameters():  # 冻结参数
                param.requires_grad = False
            teachers.append(teacher)

    rank = []
    for teacher in teachers:
        total_r = 0
        for episode in range(2):
            dead = False
            score, start_life = 0, 5
            s = ColorMat2Binary(env.reset())
            s_shadow = np.stack((s, s, s, s), axis=0)
            for _ in range(10000):
                tes = torch.unsqueeze(torch.FloatTensor(s_shadow).cuda(), 0)
                prob = teacher.pi(tes)
                a = prob.argmax()
                if a == 0:
                    real_action = 1
                elif a == 1:
                    real_action = 2
                else:
                    real_action = 3
                if dead:
                    real_action = 1
                    dead = False
                s_, r, done, info = env.step(real_action)
                s_prime = np.reshape(ColorMat2Binary(s_), (1, 84, 84))

                s_prime_shadow = np.append(s_shadow[1:, :, :], s_prime, axis=0)
                s_shadow = s_prime_shadow
                total_r += r
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                if done:
                    break
        rank.append(total_r)
        print(total_r)
    rank = softmax(np.array(rank))
    print(rank)
    return rank, teachers


def ColorMat2Binary(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return state

def train(global_model, global_r_lst):
    local_model = AC().cuda()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3000, gamma=0.8)
    alpha = 0.5
    env = gym.make('BreakoutDeterministic-v4')
    env.seed(1)
    max_step = 10000
    rank, teachers = load_teacher(env)

    for n_epi in range(max_train_ep):
        step = 0
        ep_r = 0
        dead = False
        score, start_life = 0, 5
        done = False
        s = ColorMat2Binary(env.reset())
        s_shadow = np.stack((s, s, s, s), axis=0)
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.unsqueeze(torch.FloatTensor(s_shadow).cuda(), 0))
                m = Categorical(prob)
                a = m.sample().item()
                if a == 0:
                    real_action = 1
                elif a == 1:
                    real_action = 2
                else:
                    real_action = 3
                if dead:
                    a = 0
                    real_action = 1
                    dead = False
                s_prime, r, done, info = env.step(real_action)
                s_prime = np.reshape(ColorMat2Binary(s_prime), (1, 84, 84))

                s_prime_shadow = np.append(s_shadow[1:, :, :], s_prime, axis=0)
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                ep_r += r
                step += 1
                if dead:
                    r = -1
                s_lst.append(s_shadow)
                a_lst.append([a])
                r_lst.append([r])

                s_shadow = s_prime_shadow
                if done or step >= max_step:
                    break
            s_final = torch.unsqueeze(torch.FloatTensor(s_prime_shadow).cuda(), 0)

            idx = np.random.choice(len(rank), 1, p=rank)[0]
            teacher = teachers[idx]

            R = 0.0 if done else teacher.v(s_final).item()
            s_batch, a_batch = torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda()
            values = teacher.v(s_batch).cpu().detach().numpy()
            tds = r_lst + gamma * np.append(values[1:], [[R]], axis=0) - values
            advantage = signal.lfilter([1.0], [1.0, -gamma * lam], tds[::-1])[::-1]
            advantage = torch.tensor(advantage.copy(), dtype=torch.float).cuda()

            pi = local_model.pi(s_batch)
            actions_log_probs = torch.log(pi.gather(1, a_batch) + 1e-8)
            policy_loss = -((actions_log_probs * advantage.detach()).sum())
            entropy_loss = (-torch.log(pi + 1e-8) * torch.exp(torch.log(pi + 1e-8))).sum()
            rl_loss = policy_loss - 0.01 * entropy_loss

            teacher_prob = teacher.pi(s_batch)
            KL = torch.nn.KLDivLoss()
            teach_loss = KL(pi, teacher_prob.detach()).sum()
            loss = (1-alpha) * rl_loss + alpha * teach_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param.grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        scheduler.step()
        print(n_epi, ep_r, loss.item(), alpha)
        global_r_lst.append(ep_r)
        # if n_epi > 2000:
        #     alpha = 0
        # elif n_epi > 1000:
        #     alpha = 0.01
        # elif n_epi > 50:
        #     alpha = 0.1
        if n_epi % 20 == 0:
            alpha -= 0.001
            if alpha <= 0:
                alpha = 0
        if n_epi % 100 == 0:
            np.save('data/student45.npy', np.array(global_r_lst))
        if n_epi % 100 == 0:
            torch.save(local_model, 'model/student45.pth')
    env.close()
    print(len(global_r_lst))


if __name__ == '__main__':
    global_model = AC().cuda()
    global_model.share_memory()
    global global_r_lst
    global_r_lst = []
    train(global_model, global_r_lst)
