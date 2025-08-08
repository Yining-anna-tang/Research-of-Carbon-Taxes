import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Load dataset
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv', encoding="GBK")
df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split, KFold

X = df.drop(['Y'], axis=1)
y = df['Y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# CatBoost parameters
params_cat = {
    'learning_rate': 0.01,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

# K-fold cross-validation
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

# Evaluation
from sklearn import metrics

y_pred = best_model.predict(X_test)
y_pred_list = y_pred.tolist()

mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared:", r2)

# SHAP explanation
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

labels = X_test.columns
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

# Create SHAP Explanation object
shap_explanation = shap.Explanation(
    values=shap_values[0:500, :],
    base_values=explainer.expected_value,
    data=X_test.iloc[0:500, :],
    feature_names=X_test.columns
)

# Heatmap plot
shap.plots.heatmap(shap_explanation, show=False)
plt.savefig('Y3-t2-heatmap.png', dpi=1000, format='png')
plt.show()
plt.figure()

# SHAP Force Plot for a single sample
sample_index = 1
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_test.iloc[sample_index],
    matplotlib=True,
    show=False
)
plt.savefig('WTP-t2-forceplot.png', dpi=1000, format='png')
plt.show()
plt.figure()
