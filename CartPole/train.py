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
    max_step = 8000
    for _ in range(max_step):
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
        else:
            action = agent.predict(obs)
        action_list.append(action)
        # env.render()
        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def calc_carpole_reward(reward_list):
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += reward_list[i + 1]
    return np.array(reward_list)


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
            max_step = 8000
            for _ in range(max_step):
                action = teacher.pi(torch.tensor(obs, device='cuda', dtype=torch.float)).argmax().item()
                obs, reward, done, _ = env.step(action)
                total_r += reward
                if done:
                    break
        rank.append(total_r)
    rank = softmax(np.array(rank))
    print(rank)
    return rank, teachers


def main():
    env = gym.make('CartPole-v1')
    env.seed(1)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    learning_rate = 1e-3
    alpha = 1  # teacher辅助率
    max_ep = 1200  # 总回合数

    student_total_reward = []
    rank, teachers = load_teacher(env)
    student_model = Student_Model(state_space, action_space)
    student_alg = Student_Policy(student_model, lr=learning_rate, teachers=teachers, rank=rank)
    student = Student_Agent(student_alg)

    for i in range(max_ep):  # train student
        obs_list, action_list, reward_list = run_episode(env, student)
        if i % 10 == 0:
            print("student Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

        student_total_reward.append(sum(reward_list))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_carpole_reward(reward_list)

        student.learn(batch_obs, batch_action, batch_reward, alpha=alpha)
        if i > 500:
            alpha = 0.5
        elif i > 700:
            alpha = 0.1
        elif i > 900:
            alpha = 0

    np.save('data/student20.npy', np.array(student_total_reward))


if __name__ == '__main__':
    main()
