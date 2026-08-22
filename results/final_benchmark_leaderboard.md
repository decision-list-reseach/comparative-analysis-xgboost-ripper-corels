# Final Benchmark Leaderboard

This document provides a comprehensive comparison of all trained models based on the latest 5-fold cross-validation execution.

## 1. Predictive Performance & Complexity

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 Score | Training Time (s) | Total Rules | Logical Conditions |
|-------|----------|-------------------|----------------|----------|-------------------|-------------|--------------------|
| **XGBoost** | 0.9349 | 0.8018 | 0.8237 | **0.8124** | 0.1968 | Black-box | Black-box |
| **RIPPER** | 0.8854 | 0.6825 | 0.6281 | **0.6520** | 8.2957 | 88 | 307 |
| **CORELS** | 0.8651 | 0.7112 | 0.3570 | **0.4752** | 0.0548 | 1 | 2 |

## 2. Rankings

### Predictive Leaderboard (Ranked by F1 Score)
1. **XGBoost** (0.8124)
2. **RIPPER** (0.6520)
3. **CORELS** (0.4752)

### Interpretability Leaderboard (Ranked by Minimum Logical Conditions)
1. **CORELS** (2 conditions)
2. **RIPPER** (307 conditions)
3. **XGBoost** (Black-box ensemble)

## Summary Analysis
- **Performance:** XGBoost remains a strong purely predictive model. CORELS heavily prioritizes rule compactness, heavily sacrificing Recall and therefore its overall F1 score.
- **Interpretability:** CORELS produces a provably optimal, highly interpretable rule list with only 2 conditions. RIPPER achieves better predictive performance but generates a more complex ruleset (307 conditions).
