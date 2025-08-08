import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 1.读取数据，设置变量
# 第一处修改：文件路径
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/WTP.csv',encoding = "GBK")
df = pd.DataFrame(data)

# 第二处修改：检查或修改被解释变量为Y
#
X = df.drop(['Y'],axis=1)
y = df['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# CatBoost模型参数
params_cat = {
    'learning_rate': 0.01,       # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'iterations': 1000,          # 弱学习器（决策树）的数量
    'depth': 6,                  # 决策树的深度，控制模型复杂度
    'eval_metric': 'RMSE',       # 评估指标，这里使用均方根误差（Root Mean Squared Error，简称RMSE）
    'random_seed': 42,           # 随机种子，用于重现模型的结果
    'verbose': 500               # 控制CatBoost输出信息的详细程度，每100次迭代输出一次
}

# 准备k折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

# 交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

    # 预测并计算得分
    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)  # RMSE

    scores.append(score)
    print(f'第 {fold + 1} 折 RMSE: {score}')

    # 保存得分最好的模型
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')


# 模型评估
from sklearn import metrics

# 预测、计算1、2、3指标
y_pred_four = best_model.predict(X_test)
y_pred_list = y_pred_four.tolist()
mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)
# 计算5.cc相对系数
cc = np.corrcoef(y_test, y_pred_list)[0, 1]
# 计算6.RSD相对标准差
mean_pred = np.mean(y_pred_list)
std_pred = np.std(y_pred_list)
rsd = (std_pred / mean_pred)

print("1.相对标准差 (RSD):", rsd)
print("2.相关系数 (cc):", cc)
print("3.均方根误差 (RMSE):", rmse)
print("4.均方误差 (MSE):", mse)
print("5.平均绝对误差 (MAE):", mae)
print("6.拟合优度 (R-squared):", r2)

# 模型解释
# shap解释摘要图
import shap
# 构建 shap解释器
explainer = shap.TreeExplainer(best_model)
# 计算测试集的shap值
shap_values = explainer.shap_values(X_test)


# 特征标签与蜂窝图（=解释摘要图）
labels = X_test.columns
plt.rcParams['font.family'] = 'Times new Roman'
plt.rcParams['font.serif'] = 'Times new Roman'
plt.rcParams['font.size'] = 25
shap.summary_plot(shap_values, X_test, feature_names=labels, plot_type="dot", show=False)
# 确保布局良好
plt.tight_layout()
plt.savefig('Y1-T4-1.png', dpi=1000, format='png')  # 保存图形
plt.show()  # 显示图形
plt.figure()

# 决策树
expected_value = explainer.expected_value
shap_array = explainer.shap_values(X)
shap.decision_plot(expected_value, shap_array[0:10],feature_names=list(X.columns), show=False)
plt.savefig('Y1-T4-2.png', dpi=1000, format='png')  # 保存图形
plt.show()  # 显示图形
plt.figure()









