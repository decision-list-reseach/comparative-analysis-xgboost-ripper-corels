import sys
import os
sys.path.append(os.path.abspath('../src/preprocessing'))
import pandas as pd
from pipelines import build_corels_pipeline
import yaml
import sklearn
sklearn.set_config(transform_output="default")
def load_config(p):
    with open(p, 'r') as f: return yaml.safe_load(f)
config = load_config("../src/preprocessing/config.yaml")
df = pd.read_csv("../data/ecommerce/data_ecommerce_customer_churn.csv")
X = df.drop(columns=["Churn"]).iloc[::2]
pipeline = build_corels_pipeline(config)
print("Testing full pipeline...")
res_np = pipeline.fit_transform(X)
cols = pipeline.get_feature_names_out()
print(f"Shape of res_np: {res_np.shape}, number of cols: {len(cols)}")
