import torch
import AC_20
from A3C import ActorCritic

teacher = torch.load('model/0_AC.pth')
student = AC_20.AC()

teacher_param = sum([param.nelement() for param in teacher.parameters()])
student_param = sum([param.nelement() for param in student.parameters()])
print("Number of teacher parameter: %.2fM" % (teacher_param / 1e+6))
print("Number of student parameter: %.2fM" % (student_param / 1e+6))
print('percent：%.2f' % (student_param / teacher_param))

