# External library imports with aliases
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Step 1: Load dataset and define variables
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split, KFold

X = df.drop(['Y'], axis=1)
y = df['Y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# CatBoost model parameters
params_cat = {
    'learning_rate': 0.01,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

# Set up k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

# Cross-validation loop
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

print(f'Best RMSE across folds: {best_score}')

# Model evaluation
from sklearn import metrics

y_pred = best_model.predict(X_test)
y_pred_list = y_pred.tolist()

mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

print("1. Mean Squared Error (MSE):", mse)
print("2. Root Mean Squared Error (RMSE):", rmse)
print("3. Mean Absolute Error (MAE):", mae)
print("4. R-squared:", r2)

# Model explanation using SHAP
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Set font and plot aesthetics
labels = X_test.columns
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 13

# PDP Plot 1: Feature 'eia' with interaction 'age'
shap.dependence_plot('eia', shap_values, X_test, interaction_index='age', show=False)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
plt.savefig('T1_PDP_eia_age.png', dpi=1000, format='png')
plt.show()
plt.figure()

# PDP Plot 2: Feature 'esk' with interaction 'age'
shap.dependence_plot('esk', shap_values, X_test, interaction_index='age', show=False)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
plt.savefig('T1_PDP.png', dpi=1000, format='png')
plt.show()
plt.figure()
