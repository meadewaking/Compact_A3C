import torch
import torch.nn as nn
import torch.nn.functional as F
from parl.core.torch.model import Model


class Student_Model(Model):
    def __init__(self, obs_dim, act_dim):
        super(Student_Model, self).__init__()
        hid1_size = 23
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        prob = F.softmax(self.fc2(out), dim=-1)
        return prob
