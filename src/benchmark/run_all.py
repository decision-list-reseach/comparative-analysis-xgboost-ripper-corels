import os
import subprocess
import pandas as pd
import re

# Resolve directories
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

def run_script(script_path):
    print(f"Executing {script_path}...")
    venv_python = os.path.join(project_root, ".venv_corels_310", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = "python"
        
    result = subprocess.run(
        [venv_python, script_path],
        cwd=src_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error executing {script_path}:\n{result.stderr}")
    else:
        print(f"Successfully executed {script_path}.")

def get_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df_last_5 = df.tail(5)
    return df_last_5.mean().to_dict()

def get_rule_stats(txt_path):
    stats = {'Total rules': 'N/A', 'Total logical conditions': 'N/A'}
    if not os.path.exists(txt_path):
        return stats
    with open(txt_path, 'r') as f:
        content = f.read()
    rules_match = re.search(r'Total rules:\s*(\d+)', content)
    conds_match = re.search(r'Total logical conditions:\s*(\d+)', content)
    
    if rules_match:
        stats['Total rules'] = int(rules_match.group(1))
    if conds_match:
        stats['Total logical conditions'] = int(conds_match.group(1))
    return stats

def safe_get(metrics_dict, key):
    if metrics_dict is None or key not in metrics_dict:
        return None
    return metrics_dict[key]

def format_float(val):
    if val is None: return "N/A"
    return f"{val:.4f}"

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ecommerce", choices=["ecommerce", "telco"])
    args = parser.parse_args()
    
    print(f"--- Starting Benchmark Orchestrator ({args.dataset}) ---")
    
    scripts = [
        "XGBoost/gxb_model.py",
        "RIPPER/RIPPER_model.py",
        "CORELS/train_corels.py"
    ]
    
    for s in scripts:
        print(f"Executing {s}...")
        venv_python = os.path.join(project_root, ".venv_corels_310", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = "python"
        
        result = subprocess.run(
            [venv_python, s, "--dataset", args.dataset],
            cwd=src_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error executing {s}:\n{result.stderr}")
        else:
            print(f"Successfully executed {s}.")
        
    print("\n--- Aggregating Results ---")
    
    xgb_metrics = get_metrics(os.path.join(src_dir, "XGBoost", "logs", args.dataset, "scores_from_xgboost.csv"))
    if xgb_metrics is None:
        xgb_metrics = get_metrics(os.path.join(src_dir, "XGBoost", "logs", args.dataset, "scores_from_xgb1.csv"))

    rip_metrics = get_metrics(os.path.join(src_dir, "RIPPER", "logs", args.dataset, "scores_from_ripper.csv"))
    cor_metrics = get_metrics(os.path.join(src_dir, "CORELS", "logs", args.dataset, "scores_from_corels.csv"))
    
    rip_stats = get_rule_stats(os.path.join(project_root, "results", args.dataset, "ripper_rules_test.txt"))
    if rip_stats['Total rules'] == 'N/A':
        rip_stats = get_rule_stats(os.path.join(project_root, "results", args.dataset, "ripper_rules.txt"))
        
    cor_stats = get_rule_stats(os.path.join(project_root, "results", args.dataset, "corels_rules.txt"))
    xgb_stats = {'Total rules': 'Black-box', 'Total logical conditions': 'Black-box'}
    
    # Calculate rankings dynamically
    models_f1 = [
        ("XGBoost", safe_get(xgb_metrics, 'f1')),
        ("RIPPER", safe_get(rip_metrics, 'f1')),
        ("CORELS", safe_get(cor_metrics, 'f1'))
    ]
    models_f1 = sorted([m for m in models_f1 if m[1] is not None], key=lambda x: x[1], reverse=True)
    f1_ranking_str = "\n".join([f"{i+1}. **{m[0]}** ({format_float(m[1])})" for i, m in enumerate(models_f1)])
    
    models_interp = [
        ("CORELS", cor_stats['Total logical conditions']),
        ("RIPPER", rip_stats['Total logical conditions']),
    ]
    models_interp = sorted([m for m in models_interp if isinstance(m[1], int)], key=lambda x: x[1])
    interp_ranking_str = "\n".join([f"{i+1}. **{m[0]}** ({m[1]} conditions)" for i, m in enumerate(models_interp)])
    interp_ranking_str += f"\n{len(models_interp)+1}. **XGBoost** (Black-box ensemble)"
    
    markdown_content = f"""# Final Benchmark Leaderboard ({args.dataset})

This document provides a comprehensive comparison of all trained models based on the latest 5-fold cross-validation execution.

## 1. Predictive Performance & Complexity

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 Score | Training Time (s) | Total Rules | Logical Conditions |
|-------|----------|-------------------|----------------|----------|-------------------|-------------|--------------------|
| **XGBoost** | {format_float(safe_get(xgb_metrics, 'accuracy'))} | {format_float(safe_get(xgb_metrics, 'churn_precision'))} | {format_float(safe_get(xgb_metrics, 'churn_recall'))} | **{format_float(safe_get(xgb_metrics, 'f1'))}** | {format_float(safe_get(xgb_metrics, 'time_needed'))} | {xgb_stats['Total rules']} | {xgb_stats['Total logical conditions']} |
| **RIPPER** | {format_float(safe_get(rip_metrics, 'accuracy'))} | {format_float(safe_get(rip_metrics, 'churn_precision'))} | {format_float(safe_get(rip_metrics, 'churn_recall'))} | **{format_float(safe_get(rip_metrics, 'f1'))}** | {format_float(safe_get(rip_metrics, 'time_needed'))} | {rip_stats['Total rules']} | {rip_stats['Total logical conditions']} |
| **CORELS** | {format_float(safe_get(cor_metrics, 'accuracy'))} | {format_float(safe_get(cor_metrics, 'churn_precision'))} | {format_float(safe_get(cor_metrics, 'churn_recall'))} | **{format_float(safe_get(cor_metrics, 'f1'))}** | {format_float(safe_get(cor_metrics, 'time_needed'))} | {cor_stats['Total rules']} | {cor_stats['Total logical conditions']} |

## 2. Rankings

### Predictive Leaderboard (Ranked by F1 Score)
{f1_ranking_str}

### Interpretability Leaderboard (Ranked by Minimum Logical Conditions)
{interp_ranking_str}

## Summary Analysis
- **Performance:** XGBoost remains a strong purely predictive model. CORELS heavily prioritizes rule compactness, heavily sacrificing Recall and therefore its overall F1 score.
- **Interpretability:** CORELS produces a provably optimal, highly interpretable rule list with only {cor_stats.get('Total logical conditions', 'N/A')} conditions. RIPPER achieves better predictive performance but generates a more complex ruleset ({rip_stats.get('Total logical conditions', 'N/A')} conditions).
"""

    out_path = os.path.join(project_root, "results", args.dataset, "final_benchmark_leaderboard.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(markdown_content)
        
    print(f"Successfully generated leaderboard at {out_path}")

if __name__ == "__main__":
    main()
