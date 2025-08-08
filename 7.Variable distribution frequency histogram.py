import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

sns.histplot(df['Y'], bins=10, kde=False)
plt.xlabel('Willingness to Pay')
plt.ylabel('Frequency')
plt.title('Histogram of Willingness to Pay')
plt.grid(False)
plt.savefig('Histogram-WTP.png', dpi=1000, format='png')
plt.show()
