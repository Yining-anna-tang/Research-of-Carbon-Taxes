import pandas as pd
import matplotlib.pyplot as plt

# (一）原理
# 1.创建一个示例DataFrame
# data = {'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue']}
# df = pd.DataFrame(data)

# 2.将'Color'列转换为类别类型
# df['Color'] = df['Color'].astype('category')

# 3.查看DataFrame和类别类型
# print(df)
# print(df['Color'].dtype)  # 输出: category

# （二）应用练习
# 1.读取数据，设置变量
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df = pd.DataFrame(data)

# 将'Y'列转换为类别类型
# df['Y'] = df['Y'].astype('category')

# 1️⃣获取变量Y列现在的属性（即“类别”）
# print(df['Y'].dtype)
# 2️⃣获取Y类别内容
# print(df['Y'].cat.categories)
# 3️⃣获取Y类别数据编码
# print(df['Y'].cat.codes)

# 将'age'列转换为类别类型
df['age'] = df['age'].astype('category')
# print(df['age'].dtype)
# print(df['age'].cat.categories)




