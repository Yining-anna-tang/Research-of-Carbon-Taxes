# 报错问题str

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df_raw = pd.DataFrame(data)

# 按制造商分组，并计算每个制造商的平均城市里程
df = df_raw[['Y', 'age']].groupby('age').apply(
    lambda x: x.mean())

# 现在 'age' 是作为一列存在的，如果你想要重命名它（比如改为 'age_group'）
df = df.rename(columns={'age': 'age_group'})
df['age_str'] = df['age_group'].astype(str).str.upper()

# 按城市里程排序数据
df.sort_values('Y', inplace=True)  # 按城市里程排序数据
df.reset_index(inplace=True)  # 重置索引

# 绘图
# 创建图形和坐标轴对象
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=80)

# 使用vlines绘制垂直线条，代表城市里程
ax.vlines(x=df.index, ymin=0, ymax=df.Y, color='firebrick',
          alpha=0.7, linewidth=20)

# 添加文本注释
# 在每个条形的顶部添加数值标签
for i, Y in enumerate(df.Y):
    ax.text(i, Y + 0.5, round(Y, 1), horizontalalignment='center')

# 设置标题、标签、刻度和y轴范围
ax.set_title('Bar Chart for Highway Mileage',
             fontdict={'size': 18})
ax.set(ylabel='Miles Per Gallon', ylim=(0, 30))
plt.xticks(df.index, df.age_group.str.upper(), rotation=60,
           horizontalalignment='right', fontsize=8)

# 添加补丁以为X轴标签着色
# 创建两个补丁对象，用于着色X轴标签的背景
p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1,
                       facecolor='green', transform=fig.transFigure)  # 创建绿色补丁
p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1,
                       facecolor='red', transform=fig.transFigure)  # 创建红色补丁
# 将补丁对象添加到图形上
fig.add_artist(p1)  # 添加绿色补丁
fig.add_artist(p2)  # 添加红色补丁
plt.show()