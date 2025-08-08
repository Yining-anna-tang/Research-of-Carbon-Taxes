import numpy as np
from windrose import WindroseAxes
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

# Read data and set variables
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)
wd = df['Y']
ws = df['age']

# Create windrose plot object and plot normalized bar chart
ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor="white")
ax.set_legend()
plt.show()
plt.figure()

# Create windrose plot object and plot boxplot
ax = WindroseAxes.from_ax()
ax.box(wd, ws, bins=np.arange(0, 8, 1))
ax.set_legend()
plt.show()
plt.figure()

# Create windrose plot object and plot filled contour plot
ax = WindroseAxes.from_ax()
ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)
ax.set_legend()
plt.show()
