import gym
import numpy as np
import time
import torch
from torch.distributions import Categorical
import cv2
from AC_20 import AC


def ColorMat2Binary(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    return state


agent = torch.load('model/student20.pth').cpu()
env = gym.make('SpaceInvadersDeterministic-v4')
for i in range(10):
    reward_list = []
    s = ColorMat2Binary(env.reset())
    s_shadow = np.stack((s, s, s, s), axis=0)
    while True:
        env.render()
        time.sleep(0.01)
        tes = torch.unsqueeze(torch.FloatTensor(s_shadow), 0)
        prob = agent.pi(tes)
        a = prob.argmax()
        s_, r, done, info = env.step(a)
        s_prime = np.reshape(ColorMat2Binary(s_), (1, 84, 84))

        s_prime_shadow = np.append(s_shadow[1:, :, :], s_prime, axis=0)
        s_shadow = s_prime_shadow
        reward_list.append(r)
        if done:
            break
    print(i, sum(reward_list))
