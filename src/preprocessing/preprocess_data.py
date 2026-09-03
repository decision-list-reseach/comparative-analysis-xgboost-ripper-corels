import os
import yaml
import pandas as pd
from pipelines import build_corels_pipeline
import sklearn
sklearn.set_config(transform_output="pandas")

import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="data/data_ecommerce_customer_churn_corels.csv")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    config_path = os.path.join(current_dir, args.config) if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)
    
    data_path = os.path.join(base_dir, config['dataset']['path'])
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', 'NaN', regex=True), errors='coerce')
    if 'Churn' in df.columns and set(df['Churn'].dropna().unique()).issubset({'Yes', 'No'}):
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    target_col = config['dataset']['target_column']
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print("Building CORELS pipeline...")
    pipeline = build_corels_pipeline(config)
    
    print("Fitting and transforming data...")
    # By setting transform_output="pandas", sklearn will automatically
    # track feature names through the ColumnTransformer and return a DataFrame.
    X_transformed = pipeline.fit_transform(X)
    
    # ------------------ FEATURE RENAMING LOGIC ------------------
    continuous_cols = config.get("features", {}).get("continuous", [])
    
    bin_mapping = {}
    try:
        # Extract discretizer from the pipeline
        continuous_pipeline = pipeline.named_transformers_['continuous']
        discretizer = continuous_pipeline.named_steps['discretizer']
        bin_edges = discretizer.bin_edges_
        
        for i, col in enumerate(continuous_cols):
            edges = bin_edges[i]
            for j in range(len(edges) - 1):
                # Construct interval-based names
                if j == 0:
                    label = f"{col} <= {edges[j+1]:.2f}"
                elif j == len(edges) - 2:
                    label = f"{col} > {edges[j]:.2f}"
                else:
                    label = f"{edges[j]:.2f} < {col} <= {edges[j+1]:.2f}"
                
                key = f"{col}_{float(j)}"
                bin_mapping[key] = label
    except Exception as e:
        print(f"Warning: Could not generate interval labels for continuous features: {e}")

    clean_cols = []
    for col in X_transformed.columns:
        # Strip the prefix added by ColumnTransformer (e.g. 'continuous__')
        if '__' in col:
            col = col.split('__', 1)[-1]
            
        # Rename using the bin mapping if it exists
        if col in bin_mapping:
            col = bin_mapping[col]
            
        clean_cols.append(col)
        
    X_transformed.columns = clean_cols
    # -------------------------------------------------------------
    
    # Re-attach target variable
    X_transformed[target_col] = y
    
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    print(f"Saving preprocessed binary dataset to: {output_path}")
    X_transformed.to_csv(output_path, index=False)
    
    print("Sample of processed data:")
    print(X_transformed.head())
    
    # Validation: Ensure all values are 0 or 1
    unique_values = pd.unique(X_transformed.values.ravel())
    # NaNs handled by imputer, but checking just in case
    print(f"Unique values in entire transformed matrix: {unique_values}")
    if set(unique_values).issubset({0, 1, 0.0, 1.0}):
        print("SUCCESS: Data is strictly binary!")
    else:
        print("WARNING: Non-binary values found.")
