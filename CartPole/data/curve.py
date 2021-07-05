import matplotlib.pyplot as plt
import numpy as np

teacher_total_reward = np.load('global_epr.npy')
student_45 = np.load('student45.npy')
student_20 = np.load('student20.npy')
step = 30

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(len(teacher_total_reward)), teacher_total_reward, 'blue', alpha=0.2)
plt.plot(range(len(teacher_total_reward)), np.convolve(teacher_total_reward, np.ones(step) / step)[:-step + 1], 'blue',
         label='A3C')
plt.plot(range(len(student_45)), student_45, 'red', alpha=0.2)
plt.plot(range(len(student_45)), np.convolve(student_45, np.ones(step) / step)[:-step + 1], 'red', label='45%')
plt.plot(range(len(student_20)), student_20, 'green', alpha=0.2)
plt.plot(range(len(student_20)), np.convolve(student_20, np.ones(step) / step)[:-step + 1], 'green', label='20%')
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()

print('Number of teacher parameter: 0.80k')
print('Number of student parameter: 0.35k' + ' percent：0.45')
print('Number of student parameter: 0.16k' + ' percent：0.20')
