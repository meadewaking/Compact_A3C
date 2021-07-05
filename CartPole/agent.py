import torch
import numpy as np
from parl.core.torch.agent import Agent


class Student_Agent(Agent):

    def __init__(self, algorithm):
        super(Student_Agent, self).__init__(algorithm)
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    def sample(self, obs):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs).cpu()
        prob = prob.data.numpy()
        action = np.random.choice(len(prob), 1, p=prob)[0]
        return action

    def predict(self, obs):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs)
        _, action = prob.max(-1)
        return action.item()

    def learn(self, obs, action, reward, alpha):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)

        loss = self.alg.learn(obs, action, reward, alpha)
        return loss.item()
