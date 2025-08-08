import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning/Y2_top.csv', encoding ="GBK")
from sklearn.model_selection import train_test_split


X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)


model_xgb = XGBRegressor(eval_metric='rmse', random_state=8)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


kfold = KFold(n_splits=5, shuffle=True, random_state=8)


grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, scoring='neg_mean_squared_error',
                           cv=kfold, verbose=1, n_jobs=-1)


grid_search.fit(X_train, y_train)

xgboost = grid_search.best_estimator_
import sys
import os


y_pred = xgboost.predict(X_test)

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("📈：")
print(f"R² Score:         {r2:.4f}")
print(f"MSE:    {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")



sys.stdout = open(os.devnull, 'w')
import shap
explainer = shap.TreeExplainer(xgboost)

shap_values_numpy = explainer.shap_values(X_test)

shap_values_Explanation = explainer(X_test)

plt.figure(figsize=(10, 5), dpi=1200)
shap.plots.bar(shap_values_Explanation, show=False, max_display=10)
plt.savefig("16.2-1(Y-WTP).pdf", format='pdf',bbox_inches='tight', dpi=1200)
plt.tight_layout()
plt.show()



shap_values_df = pd.DataFrame(shap_values_numpy, columns=X_test.columns)

abs_shap_values_mean = shap_values_df.abs().mean()

total_mean = abs_shap_values_mean.sum()
shap_values_percentage = (abs_shap_values_mean / total_mean) * 100


sorted_features = abs_shap_values_mean.sort_values(ascending=True)
sorted_percentage = shap_values_percentage[sorted_features.index]


top_n = 15
sorted_features_top = sorted_features.tail(top_n)
sorted_percentage_top = sorted_percentage[sorted_features_top.index]

plt.figure(figsize=(10, 6), dpi=300)
bars = plt.barh(
    sorted_features_top.index,
    sorted_features_top.values,
    color='skyblue',
    edgecolor='black'
)


for bar, perc in zip(bars, sorted_percentage_top):
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.4f}\n({perc:.2f}%)',
        va='center',
        fontsize=10
    )


plt.xlim(0, sorted_features_top.max() * 1.3)

plt.xlabel(" ")
plt.title(" ")
plt.tight_layout()
plt.savefig("16.2-2 (WTP).pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()



