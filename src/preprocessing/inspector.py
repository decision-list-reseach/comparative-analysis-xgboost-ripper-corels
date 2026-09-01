import pandas as pd
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_inspection_report(df, config, output_path):
    report = ["# Dataset Inspection Report\n"]
    
    report.append(f"**Total rows:** {len(df)}")
    report.append(f"**Total columns:** {len(df.columns)}\n")
    
    report.append("## Column Analysis\n")
    
    features = config.get("features", {})
    all_configured_features = {f: cat for cat, feats in features.items() for f in feats}
    
    for col in df.columns:
        if col == config.get("dataset", {}).get("target_column"):
            continue
            
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isnull().sum()
        semantic_type = all_configured_features.get(col, "Unknown")
        
        report.append(f"### {col}")
        report.append(f"- **Data Type:** {dtype}")
        report.append(f"- **Unique Values:** {n_unique}")
        report.append(f"- **Missing Values:** {n_missing} ({(n_missing/len(df))*100:.2f}%)")
        report.append(f"- **Semantic Type (from config):** {semantic_type}")
        
        if semantic_type == "binary":
            report.append("- **Recommended Transformation:** Passthrough")
        elif semantic_type == "nominal":
            report.append("- **Recommended Transformation:** One-Hot Encoding")
        elif semantic_type == "ordinal":
            report.append("- **Recommended Transformation:** Step/Cumulative Encoding")
        elif semantic_type == "continuous":
            report.append(f"- **Recommended Transformation:** Quantile Discretization (n_bins={config.get('discretization', {}).get('n_bins', 4)}) + One-Hot Encoding")
        report.append("")
        
    with open(output_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"Inspection report generated at: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    config_path = os.path.join(current_dir, "config.yaml")
    config = load_config(config_path)
    
    data_path = os.path.join(base_dir, config['dataset']['path'])
    df = pd.read_csv(data_path)
    
    output_path = os.path.join(current_dir, "report.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generate_inspection_report(df, config, output_path)
