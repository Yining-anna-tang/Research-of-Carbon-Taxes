import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set(font='Times New Roman', font_scale=0.8, style="darkgrid")

data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

sns.histplot(data=df, x="Y", color="skyblue", label="Willingness to Pay", kde=True)
sns.histplot(data=df, x="age", color="red", label="Age", kde=True)

plt.legend()
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Distribution of Willingness to Pay and Age")
plt.tight_layout()
plt.show()
