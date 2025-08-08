import pandas as pd
import matplotlib.pyplot as plt
from PyALE import ale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

# Configure plot settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/Y2_top 15.csv')

# Split features and target
X = df.drop(['Y'], axis=1)
y = df['Y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Compute ALE for single feature
ale_single = ale(X=X_test, model=rf_model, feature=["UFSC"])

# Compute ALE for interaction between two features with fine grid
ale_interaction = ale(X=X_test, model=rf_model, feature=["UFSC", "UGCCI"], grid_size=100)

# Save the plot as high-resolution PDF
plt.savefig("26.Y2-UFSC+UGCCI-ALE-Interaction.pdf", format='pdf', bbox_inches='tight', dpi=1200)
