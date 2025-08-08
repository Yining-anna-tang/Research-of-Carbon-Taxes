import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df_raw = pd.DataFrame(data)

df = df_raw[['Y', 'age']].groupby('age').mean()
df.sort_values('Y', inplace=True)
df.reset_index(inplace=True)
df = df.rename(columns={'age': 'age_group'})
df['age_str'] = df['age_group'].astype(str).str.upper()

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=80)
ax.vlines(x=df.index, ymin=0, ymax=df.Y, color='firebrick', alpha=0.7, linewidth=20)

for i, Y in enumerate(df.Y):
    ax.text(i, Y + 0.5, round(Y, 1), horizontalalignment='center')

ax.set_title('Bar Chart for Highway Mileage', fontdict={'size': 18})
ax.set(ylabel='Miles Per Gallon', ylim=(0, 30))
plt.xticks(df.index, df.age_str, rotation=60, horizontalalignment='right', fontsize=8)

p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
fig.add_artist(p1)
fig.add_artist(p2)
plt.show()
