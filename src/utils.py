from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

def save_scores(file_name, current_dir, y_test, preds, time_needed, dataset="ecommerce"):
    params_and_metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "churn_precision": precision_score(y_test, preds, pos_label=1),
        "churn_recall": recall_score(y_test, preds, pos_label=1),
        "f1": f1_score(y_test, preds),
        "time_needed": (time_needed),
    }

    df_current_run = pd.DataFrame([params_and_metrics])
    
    logs_dir = os.path.join(current_dir, 'logs', dataset)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f'{file_name}.csv')

    if not os.path.isfile(log_file):
        df_current_run.to_csv(log_file, index=False)
    else:
        df_current_run.to_csv(log_file, mode='a', header=False, index=False)