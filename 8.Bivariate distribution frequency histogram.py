import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set(font='Times New Roman', font_scale=0.8, style="darkgrid") # 解决Seaborn中文显示问题

# 导入数据
# 1.读取数据，设置变量
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df = pd.DataFrame(data)

sns.histplot(data=df, x="Y", color="skyblue", label="Y", kde=True)
sns.histplot(data=df, x="age", color="red", label="age", kde=True)

plt.legend()
plt.show()
