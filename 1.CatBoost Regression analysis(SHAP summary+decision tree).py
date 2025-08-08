import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

params_cat = {
    'learning_rate': 0.01,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)

    scores.append(score)
    print(f'Fold {fold + 1} RMSE: {score}')

    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')

from sklearn import metrics

y_pred_four = best_model.predict(X_test)
y_pred_list = y_pred_four.tolist()
mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)
cc = np.corrcoef(y_test, y_pred_list)[0, 1]
mean_pred = np.mean(y_pred_list)
std_pred = np.std(y_pred_list)
rsd = (std_pred / mean_pred)

print("1. Relative Standard Deviation (RSD):", rsd)
print("2. Correlation Coefficient (cc):", cc)
print("3. Root Mean Squared Error (RMSE):", rmse)
print("4. Mean Squared Error (MSE):", mse)
print("5. Mean Absolute Error (MAE):", mae)
print("6. R-squared:", r2)

import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

labels = X_test.columns
plt.rcParams['font.family'] = 'Times new Roman'
plt.rcParams['font.serif'] = 'Times new Roman'
plt.rcParams['font.size'] = 25
shap.summary_plot(shap_values, X_test, feature_names=labels, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('Y1-T4-1.png', dpi=1000, format='png')
plt.show()
plt.figure()

expected_value = explainer.expected_value
shap_array = explainer.shap_values(X)
shap.decision_plot(expected_value, shap_array[0:10], feature_names=list(X.columns), show=False)
plt.savefig('Y1-T4-2.png', dpi=1000, format='png')
plt.show()
plt.figure()
