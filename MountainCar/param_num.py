import torch
from model import Student_Model
import gym
from a3c import ActorCritic

env = gym.make('CartPole-v1')
env.seed(1)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

teacher = torch.load('model/0_AC.pth')
student = Student_Model(state_space, action_space)

teacher_param = sum([param.nelement() for param in teacher.parameters()])
student_param = sum([param.nelement() for param in student.parameters()])
print("Number of teacher parameter: %.2fk" % (teacher_param / 1e+3))
print("Number of student parameter: %.2fk" % (student_param / 1e+3))
print('percent：%.2f' % (student_param / teacher_param))

