import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif']=['kaiti']
add_str = "reward"
data_list = np.load("plot_data/"+add_str+"_0410_2.npy")

num = len (data_list)
x = range(num)


# 创建图形和子图
fig, ax1 = plt.subplots()

# 绘制第一个数组的曲线（左y轴）
ax1.plot(x, data_list, 'y-')
ax1.set_xlabel('训练轮次')
ax1.set_ylabel('reward', color='y')


tick_font = font_manager.FontProperties(family='DejaVu Sans', size=7.0)
for labelx  in ax1.get_xticklabels():
    labelx.set_fontproperties(tick_font) #设置 x轴刻度字体
for labely in ax1.get_yticklabels():
    labely.set_fontproperties(tick_font) #设置 y轴刻度字体

plt.title("强化学习训练平均奖励曲线图")
plt.show()