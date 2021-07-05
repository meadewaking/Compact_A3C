import torch
import AC
from a3c import ActorCritic

teacher = torch.load('model/0_AC.pth')
student = AC.AC()

teacher_param = sum([param.nelement() for param in teacher.parameters()])
student_param = sum([param.nelement() for param in student.parameters()])
print("Number of teacher parameter: %.2fM" % (teacher_param / 1e+6))
print("Number of student parameter: %.2fM" % (student_param / 1e+6))
print('percent：%.2f' % (student_param / teacher_param))

