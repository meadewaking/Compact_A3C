import matplotlib.pyplot as plt
import numpy as np

teacher_total_reward = np.load('global_epr.npy')
student_45 = np.load('student45.npy')
student_20 = np.load('student20.npy')
step = 30

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(len(teacher_total_reward)), np.clip(teacher_total_reward, -20000, 20000), 'blue', alpha=0.2)
plt.plot(range(len(teacher_total_reward) - step),
         np.clip(np.convolve(teacher_total_reward, np.ones(step) / step)[step:-step + 1], -20000, 20000), 'blue',
         label='A3C')
plt.plot(range(len(student_45)), np.clip(student_45, -20000, 20000), 'red', alpha=0.2)
plt.plot(range(len(student_45) - step),
         np.clip(np.convolve(student_45, np.ones(step) / step)[step:-step + 1], -20000, 20000), 'red',
         label='45%')
plt.plot(range(len(student_20)), np.clip(student_20, -20000, 20000), 'green', alpha=0.2)
plt.plot(range(len(student_20) - step),
         np.clip(np.convolve(student_20, np.ones(step) / step)[step:-step + 1], -20000, 20000), 'green',
         label='20%')
plt.ylabel("episode_reward")
plt.xlabel("episode")
plt.legend()
plt.show()

print('Number of teacher parameter: 0.80k')
print('Number of student parameter: 0.35k' + ' percent：0.45')
print('Number of student parameter: 0.16k' + ' percent：0.20')
