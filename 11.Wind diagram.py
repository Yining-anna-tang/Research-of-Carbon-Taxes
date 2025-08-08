import numpy as np
from windrose import WindroseAxes
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

# 生成随机的风速和风向数据
# 1.读取数据，设置变量
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df = pd.DataFrame(data)
wd = df['Y']
ws = df['age']

# 创建风玫瑰图对象，并绘制归一化柱状图
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor="white")  # 绘制柱状图
ax.set_legend()  # 添加图例
plt.show()
plt.figure()

# 创建风玫瑰图对象，并绘制箱线图
ax = WindroseAxes.from_ax()
ax.box(wd, ws, bins=np.arange(0, 8, 1))  # 绘制箱线图
ax.set_legend()  # 添加图例
plt.show()
plt.figure()

# 创建风玫瑰图对象，并绘制填充等高线图
ax = WindroseAxes.from_ax()
ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)  # 绘制填充等高线图
ax.set_legend()  # 添加图例
plt.show()