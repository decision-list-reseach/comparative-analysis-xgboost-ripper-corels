import wittgenstein as lw
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')

df = pd.read_csv(csv_path)
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])


imputer = SimpleImputer(strategy='median')
smote = SMOTE(random_state=42)
ripper = lw.RIPPER(random_state=42)

pipeline = ImbPipeline(steps=[
    ('imputer', imputer),
    ('smote', smote),
    ('ripper', ripper)
])

ripper_param_grid = {
    'ripper__k': [1, 2],
    'ripper__prune_size': [0.1, 0.2, 0.33],
    'ripper__dl_allowance': [32, 64, 128]
}

ripper_grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=ripper_param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1,
    error_score='raise'
)

y = df['Churn']
X = df.drop('Churn', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

print("Starting RIPPER grid search...")
ripper_grid_search.fit(X_train, y_train)

best_model = ripper_grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


print(f"Best RIPPER Parameters: {ripper_grid_search.best_params_}")
print(f"Training F1 Score: {f1_score(y_train, y_train_pred):.4f}")
print(f"Testing F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
print("\nExtracted Rules from Best RIPPER Model:")
best_model.named_steps['ripper'].out_model()