# Final Benchmark Leaderboard (telco)

This document provides a comprehensive comparison of all trained models based on the latest 5-fold cross-validation execution.

## 1. Predictive Performance & Complexity

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 Score | Training Time (s) | Total Rules | Logical Conditions |
|-------|----------|-------------------|----------------|----------|-------------------|-------------|--------------------|
| **XGBoost** | 0.7546 | 0.5278 | 0.7154 | **0.6074** | 0.2264 | Black-box | Black-box |
| **RIPPER** | 0.7863 | 0.6352 | 0.4661 | **0.5331** | 12.6456 | 36 | 240 |
| **CORELS** | 0.7822 | 0.7300 | 0.2846 | **0.4095** | 0.1132 | 1 | 2 |

## 2. Rankings

### Predictive Leaderboard (Ranked by F1 Score)
1. **XGBoost** (0.6074)
2. **RIPPER** (0.5331)
3. **CORELS** (0.4095)

### Interpretability Leaderboard (Ranked by Minimum Logical Conditions)
1. **CORELS** (2 conditions)
2. **RIPPER** (240 conditions)
3. **XGBoost** (Black-box ensemble)

## Summary Analysis
- **Performance:** XGBoost remains a strong purely predictive model. CORELS heavily prioritizes rule compactness, heavily sacrificing Recall and therefore its overall F1 score.
- **Interpretability:** CORELS produces a provably optimal, highly interpretable rule list with only 2 conditions. RIPPER achieves better predictive performance but generates a more complex ruleset (240 conditions).
