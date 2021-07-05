import gym
import numpy as np
import time
import torch
from torch.distributions import Categorical
import cv2
from AC import AC


def ColorMat2Binary(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return state


agent = torch.load('model/student45.pth').cpu()
env = gym.make('BreakoutDeterministic-v4')
# env = gym.make('Breakout-v4')
# env = gym.make('BreakoutNoFrameskip-v4')
env.seed(1)
for i in range(10):
    reward_list = []
    dead = False
    score, start_life = 0, 5
    s = ColorMat2Binary(env.reset())
    s_shadow = np.stack((s, s, s, s), axis=0)
    while True:
        env.render()
        # time.sleep(0.01)
        tes = torch.unsqueeze(torch.FloatTensor(s_shadow), 0)
        prob = agent.pi(tes)
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
        reward_list.append(r)
        if start_life > info['ale.lives']:
            dead = True
            start_life = info['ale.lives']
        if done:
            break
    print(i, sum(reward_list))
