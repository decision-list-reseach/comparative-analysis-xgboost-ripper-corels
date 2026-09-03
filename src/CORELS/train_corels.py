import os
import sys
import time
import pandas as pd
import numpy as np
import json
import sklearn
sklearn.set_config(transform_output="default")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from corels import CorelsClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import utils    

import argparse
from sklearn.model_selection import StratifiedKFold

# Import the preprocessing builder
sys.path.append(os.path.join(parent_dir, 'preprocessing'))
from pipelines import build_corels_pipeline
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def rename_features(X_transformed, bin_edges, continuous_cols):
    bin_mapping = {}
    try:
        for i, col in enumerate(continuous_cols):
            edges = bin_edges[i]
            for j in range(len(edges) - 1):
                if j == 0:
                    label = f"{col} <= {edges[j+1]:.2f}"
                elif j == len(edges) - 2:
                    label = f"{col} > {edges[j]:.2f}"
                else:
                    label = f"{edges[j]:.2f} < {col} <= {edges[j+1]:.2f}"
                key = f"{col}_{float(j)}"
                bin_mapping[key] = label
    except Exception as e:
        pass
    
    clean_cols = []
    for col in X_transformed.columns:
        if '__' in col:
            col = col.split('__', 1)[-1]
        if col in bin_mapping:
            col = bin_mapping[col]
        clean_cols.append(col)
    X_transformed.columns = clean_cols
    return X_transformed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ecommerce", choices=["ecommerce", "telco"])
    args = parser.parse_args()
    
    if args.dataset == "ecommerce":
        csv_path = os.path.join(project_root, 'data', 'ecommerce', 'data_ecommerce_customer_churn.csv')
        config_path = os.path.join(project_root, 'src', 'preprocessing', 'config.yaml')
    else:
        csv_path = os.path.join(project_root, 'data', 'telco', 'TEST_telco_customer_churn.csv')
        config_path = os.path.join(project_root, 'tests', 'configs', 'config_telco.yaml')
        
    df = pd.read_csv(csv_path)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', 'NaN', regex=True), errors='coerce')
    if 'Churn' in df.columns and set(df['Churn'].dropna().unique()).issubset({'Yes', 'No'}):
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    config = load_config(config_path)
    target_col = config['dataset']['target_column']
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    metrics_list = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"--- CORELS 5-Fold Stratified Cross-Validation ({args.dataset}) ---")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/5 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Build and fit preprocessing strictly on training data
        pipeline = build_corels_pipeline(config)
        X_train_trans_np = pipeline.fit_transform(X_train)
        X_test_trans_np = pipeline.transform(X_test)
        
        feature_names = pipeline.get_feature_names_out()
        X_train_trans = pd.DataFrame(X_train_trans_np, columns=feature_names)
        X_test_trans = pd.DataFrame(X_test_trans_np, columns=feature_names)
        
        # Rename features for readability
        continuous_cols = config.get("features", {}).get("continuous", [])
        discretizer = pipeline.named_transformers_['continuous'].named_steps['discretizer']
        bin_edges = discretizer.bin_edges_
        
        X_train_trans = rename_features(X_train_trans, bin_edges, continuous_cols)
        X_test_trans = rename_features(X_test_trans, bin_edges, continuous_cols)
        
        corels_clf = CorelsClassifier(c=0.01, n_iter=10000, verbosity=[], max_card=2)
        
        X_train_dense = np.asarray(X_train_trans.values, dtype=np.uint8)
        y_train_dense = np.asarray(y_train.values, dtype=np.uint8)
        X_test_dense = np.asarray(X_test_trans.values, dtype=np.uint8)
        
        start = time.perf_counter()
        corels_clf.fit(X_train_dense, y_train_dense, features=X_train_trans.columns.tolist(), prediction_name="Churn")
        end = time.perf_counter()
        time_needed = end - start
        
        preds = corels_clf.predict(X_test_dense)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label=1)
        rec = recall_score(y_test, preds, pos_label=1)
        f1 = f1_score(y_test, preds)
        
        metrics_list.append({
            "accuracy": acc,
            "churn_precision": prec,
            "churn_recall": rec,
            "f1": f1,
            "time_needed": time_needed
        })
        
        utils.save_scores("scores_from_corels", current_dir, y_test, preds, time_needed, dataset=args.dataset)

    print("\nAveraged metrics from 5 results:")
    df_metrics = pd.DataFrame(metrics_list)
    print(df_metrics.mean().to_string())
    
    rule_list_str = corels_clf.rl()
    print("\n### Provably Optimal Rule List (Last Run) ###")
    print(rule_list_str)
    
    # Save Rule List
    results_dir = os.path.join(project_root, 'results', args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    rules_path = os.path.join(results_dir, 'corels_rules.txt')
    
    # Calculate stats
    lines = str(rule_list_str).strip().split('\n')
    total_rules = 0
    total_conds = 0
    for line in lines:
        if line.startswith('if ') or line.startswith('else if '):
            total_rules += 1
            conds_in_rule = line.count('&&') + 1
            total_conds += conds_in_rule
            
    avg_conds = total_conds / total_rules if total_rules > 0 else 0
    
    header_block = f"""CORELS Rule List
================

Parameters:
c = {corels_clf.c}
max_card = {corels_clf.max_card}
policy = {corels_clf.policy}
n_iter = {corels_clf.n_iter}

================================
Rule List
================================

"""

    stats_block = f"\n================================\n\nTotal rules: {total_rules}\n\nAverage conditions per rule: {avg_conds:.2f}\n\nTotal logical conditions: {total_conds}\n"
    final_output = header_block + str(rule_list_str) + stats_block

    with open(rules_path, "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    main()
