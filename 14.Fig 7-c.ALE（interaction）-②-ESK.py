import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyALE import ale

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/Y2_top 15.csv')
from sklearn.model_selection import train_test_split, KFold

from sklearn.model_selection import train_test_split

X = df.drop(['Y'],axis=1)
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(X,  y,  test_size=0.3,  random_state=42)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)


ale_eff = ale(
    X=X_test,            
    model=rf_model,             
    feature=["ESK"]      
)

ale_eff = ale(X=X_test, model=rf_model, feature=["ESK", "UGCCI"], grid_size=100)
plt.savefig("26.Y2-ESK+UGCCI-离散特征交互图.pdf", format='pdf',bbox_inches='tight',dpi=1200)
