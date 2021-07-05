import torch
import numpy as np
from parl.core.torch.algorithm import Algorithm


class Student_Policy(Algorithm):
    def __init__(self, model, lr, teachers, rank):
        assert isinstance(lr, float)

        self.model = model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.teachers = teachers
        self.rank = rank

    def predict(self, obs):
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward, alpha):

        prob = self.model(obs)

        log_prob = torch.distributions.Categorical(prob).log_prob(action)

        idx = np.random.choice(len(self.rank), 1, p=self.rank)[0]
        teacher = self.teachers[idx]
        teacher_prob = teacher.pi(obs)
        KL = torch.nn.KLDivLoss()

        rl_loss = torch.mean(-1 * log_prob * reward)
        teach_loss = KL(prob, teacher_prob)  # 策略蒸馏

        # KL = torch.nn.KLDivLoss()
        # for i in range(len(self.rank)):
        #     teacher_prob = self.teachers[i].pi(obs)
        #     if i == 0:
        #         teach_loss = self.rank[i] * KL(prob, teacher_prob)
        #     else:
        #         teach_loss += self.rank[i] * KL(prob, teacher_prob)

        loss = (1 - alpha) * rl_loss + teach_loss * alpha

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
