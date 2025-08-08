import pandas as pd
import matplotlib.pyplot as plt

# Read data and set variables
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

# Convert 'Y' column to category type
# df['Y'] = df['Y'].astype('category')

# 1️⃣ Get the current dtype of column 'Y' (should be 'category')
# print(df['Y'].dtype)

# 2️⃣ Get the categories of 'Y'
# print(df['Y'].cat.categories)

# 3️⃣ Get the category codes of 'Y'
# print(df['Y'].cat.codes)

# Convert 'age' column to category type
df['age'] = df['age'].astype('category')
# print(df['age'].dtype)
# print(df['age'].cat.categories)
