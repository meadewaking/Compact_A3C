import gym
import random
import time
import numpy as np
import torch
from a3c import ActorCritic

agent = torch.load('model/1_AC.pth')
env = gym.make('MountainCar-v0')
env.seed(1)
obs_list = []
for i in range(10):
    reward_list = []
    obs_list = []
    s = env.reset()
    while True:
        env.render()
        # time.sleep(0.02)
        obs_list.append(s)
        a = agent.pi(torch.from_numpy(s).float()).argmax().item()
        s_, r, done, info = env.step(a)
        s = s_
        reward_list.append(r)
        if done:
            break
    print(i, sum(reward_list))

