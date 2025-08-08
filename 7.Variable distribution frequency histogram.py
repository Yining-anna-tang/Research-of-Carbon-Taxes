# 命名简写外部库
import pandas as pd
# 导入numpy库，简称为np
import numpy as np
# 导入seaborn库，简称为sns
import seaborn as sns
# 导入matplotlib.pyplot模块，简称为plt
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 1.读取数据，设置变量
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df = pd.DataFrame(data)

# 使用seaborn绘制age的频率分布图
# 取颜色
sns.histplot(df['Y'], bins=10, kde=False)  # kde=False表示不绘制核密度估计曲线
plt.xlabel('          ')
plt.ylabel('          ')
plt.title('          ')
plt.grid(False)  # grid=False表示不显示网格背景
plt.savefig('Y4-频率直方图2-Y.png', dpi=1000, format='png')
plt.show()