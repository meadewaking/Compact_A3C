import gym
import numpy as np
from model import Student_Model
from algorithm import Student_Policy
from agent import Student_Agent
import torch
import os
from a3c import ActorCritic


def run_episode(env, agent, train_or_test='train'):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    max_step = 20000
    for _ in range(max_step):
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
        else:
            action = agent.predict(obs)
        action_list.append(action)
        obs, reward, done, _ = env.step(action)
        if done:
            reward = 10
        elif obs[0] >= 0.4:
            reward = 1
        elif obs[0] >= -0.1:
            reward = 0
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def softmax(x):
    x = (x - np.mean(x))/(np.std(x) + 1e-6)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


def load_teacher(env):
    pth_list = os.listdir('model/')
    teachers = []
    for file in pth_list:
        teacher = torch.load('model/' + file).cuda()
        for param in teacher.parameters():      #冻结参数
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


def calc_reward_to_go(reward_list):
    discounted_ep_rs = np.zeros_like(reward_list)
    running_add = 0
    for t in reversed(range(0, len(reward_list))):
        running_add = running_add * 0.99 + reward_list[t]
        discounted_ep_rs[t] = running_add

    # normalize episode rewards
    #discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= 10.0
    return discounted_ep_rs

def main():
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(1)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    learning_rate = 1e-3
    alpha = 0.5  # teacher辅助率
    max_ep = 800  # 总回合数
    loss = 0
    student_total_reward = []
    rank, teachers = load_teacher(env)
    student_model = Student_Model(state_space, action_space)
    student_alg = Student_Policy(student_model, lr=learning_rate, teachers=teachers, rank=rank)
    student = Student_Agent(student_alg)

    for i in range(max_ep):  # train student
        obs = env.reset()
        done = False
        ep_r = 0
        for j in range(50):
            obs_list, action_list, reward_list = [], [], []
            for _ in range(20):
                obs_list.append(obs)
                action = student.sample(obs)
                action_list.append(action)
                obs, reward, done, _ = env.step(action)
                ep_r += reward
                if done:
                    reward = 10
                elif obs[0] >= 0.3:
                    reward = 1
                elif obs[0] >= -0.2:
                    reward = 0
                if ep_r <= -998:
                    reward = -10
                reward_list.append(reward/1.0)

                if done:
                    break
            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list)
            loss = student.learn(batch_obs, batch_action, batch_reward, alpha=alpha)
            if done:
                break
        if i > 5:
            alpha = 0.1
        elif i > 50:
            alpha = 0.01
        elif i > 100:
            alpha = 0
        if i % 1 == 0:
            print("student Episode {}, Reward Sum {}.".format(i, ep_r), loss)

        student_total_reward.append(ep_r)



    np.save('data/student45.npy', np.array(student_total_reward))


if __name__ == '__main__':
    main()
