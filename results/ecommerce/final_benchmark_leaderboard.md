# Final Benchmark Leaderboard (ecommerce)

This document provides a comprehensive comparison of all trained models based on the latest 5-fold cross-validation execution.

## 1. Predictive Performance & Complexity

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 Score | Training Time (s) | Total Rules | Logical Conditions |
|-------|----------|-------------------|----------------|----------|-------------------|-------------|--------------------|
| **XGBoost** | 0.9312 | 0.7703 | 0.8546 | **0.8094** | 0.1948 | Black-box | Black-box |
| **RIPPER** | 0.8790 | 0.6931 | 0.5595 | **0.5917** | 6.8953 | 80 | 278 |
| **CORELS** | 0.8632 | 0.7033 | 0.3487 | **0.4649** | 0.0560 | 1 | 2 |

## 2. Rankings

### Predictive Leaderboard (Ranked by F1 Score)
1. **XGBoost** (0.8094)
2. **RIPPER** (0.5917)
3. **CORELS** (0.4649)

### Interpretability Leaderboard (Ranked by Minimum Logical Conditions)
1. **CORELS** (2 conditions)
2. **RIPPER** (278 conditions)
3. **XGBoost** (Black-box ensemble)

## Summary Analysis
- **Performance:** XGBoost remains a strong purely predictive model. CORELS heavily prioritizes rule compactness, heavily sacrificing Recall and therefore its overall F1 score.
- **Interpretability:** CORELS produces a provably optimal, highly interpretable rule list with only 2 conditions. RIPPER achieves better predictive performance but generates a more complex ruleset (278 conditions).
