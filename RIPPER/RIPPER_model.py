import os
import re
import sys
import time

import numpy as np
import pandas as pd
import wittgenstein as lw
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Paths & shared utilities
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import save_scores

# ---------------------------------------------------------------------------
# Data loading & encoding
# ---------------------------------------------------------------------------
csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')
df = pd.read_csv(csv_path)
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])

y = df['Churn']
X = df.drop('Churn', axis=1)

# ---------------------------------------------------------------------------
# 5-fold stratified cross-validation
# Best RIPPER parameters: k=1, prune_size=0.33, dl_allowance=128
# ---------------------------------------------------------------------------
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipeline = None  # will hold the last fold's fitted pipeline

print("=" * 60)
print("RIPPER  —  5-Fold Stratified Cross-Validation")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fresh pipeline per fold to avoid state bleed between folds
    pipeline = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('smote',   SMOTE(random_state=42)),
        ('ripper',  lw.RIPPER(k=1, prune_size=0.33, dl_allowance=128, random_state=42))
    ])

    print(f"\n  Fold {fold}/5 — training...")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    end = time.perf_counter()
    time_needed = end - start

    preds      = pipeline.predict(X_test)
    train_pred = pipeline.predict(X_train)

    print(f"  Fold {fold}/5 | time: {time_needed:.2f}s"
          f" | train F1: {f1_score(y_train, train_pred):.4f}"
          f" | test F1: {f1_score(y_test, preds):.4f}")

    save_scores("scores_from_ripper", current_dir, y_test, preds, time_needed)

# ---------------------------------------------------------------------------
# Rule export — from the last fold's fitted model
# ---------------------------------------------------------------------------
results_dir     = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
ripper_rules_path = os.path.join(results_dir, 'ripper_rules_test.txt')

ripper_model_step = pipeline.named_steps['ripper']
rule_str = str(ripper_model_step.ruleset_)

# Replace column indices with human-readable names
for idx, col_name in reversed(list(enumerate(X.columns))):
    rule_str = re.sub(rf'\b{idx}=', f'{col_name}=', rule_str)

# Reformat for readability
if rule_str.startswith('[['):
    rule_str = rule_str[2:]
if rule_str.endswith(']]'):
    rule_str = rule_str[:-2]

rules = rule_str.split('] V [')
formatted_rules = []

for i, r in enumerate(rules, 1):
    cond_str = r.replace('^', '\nAND ')
    # Normalise spacing around AND
    cond_str = cond_str.replace(' \nAND  ', '\nAND ')
    cond_str = cond_str.replace(' \nAND ',  '\nAND ')
    cond_str = cond_str.replace('\nAND  ',  '\nAND ')
    # Format operators
    cond_str = cond_str.replace('=>', ' >= ').replace('=<', ' <= ')
    cond_str = re.sub(r'=(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)', r' ∈ [\1, \2]', cond_str)
    block = f'Rule {i}\nIF\n{cond_str}\n\nTHEN Churn\n\n------------------------\n'
    formatted_rules.append(block)

# Robust counting from internal representation
total_rules = len(ripper_model_step.ruleset_.rules)
total_conds = sum(len(rule.conds) for rule in ripper_model_step.ruleset_.rules)
avg_conds   = total_conds / total_rules if total_rules > 0 else 0

header_block = f"""RIPPER Rule Set
================

Model parameters:
k            = {ripper_model_step.k}
prune_size   = {ripper_model_step.prune_size}
dl_allowance = {ripper_model_step.dl_allowance}
random_state = {ripper_model_step.random_state}

================================
Rules
================================

"""

stats_block = (
    f"\n================================\n\n"
    f"Total rules: {total_rules}\n\n"
    f"Average conditions per rule: {avg_conds:.2f}\n\n"
    f"Total logical conditions: {total_conds}\n"
)

final_output = header_block + '\n'.join(formatted_rules) + stats_block

with open(ripper_rules_path, 'w', encoding="utf-8") as f:
    f.write(final_output)

print(f"\nRule list saved to: {ripper_rules_path}")