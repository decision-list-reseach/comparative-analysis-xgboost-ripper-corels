import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import os

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'scale_pos_weight': [3,4,5]
}


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')

df = pd.read_csv(csv_path)
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])

y = df['Churn']
X = df.drop('Churn', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

base_xgb = XGBClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_xgb, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1',
    verbose=1
)

print("Starting grid search...")
grid_search.fit(X_train, y_train)

print(f"The Best Parameters are: {grid_search.best_params_}")