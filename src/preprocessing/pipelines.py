import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer

try:
    from .encoders import StepEncoder
except ImportError:
    from encoders import StepEncoder

def make_passthrough_transformer():
    return FunctionTransformer(lambda x: x, feature_names_out='one-to-one')

def build_corels_pipeline(config):
    """
    Builds a robust, non-leaking Scikit-Learn preprocessing pipeline for CORELS.
    """
    features = config.get("features", {})
    binary_cols = features.get("binary", [])
    nominal_cols = features.get("nominal", [])
    ordinal_cols = features.get("ordinal", [])
    continuous_cols = features.get("continuous", [])
    
    n_bins = config.get("discretization", {}).get("n_bins", 4)
    strategy = config.get("discretization", {}).get("strategy", "quantile")
    
    # 1. Continuous: Median Imputation + Missing Indicator -> Quantile Discretization -> OneHot
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('discretizer', KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy=strategy, subsample=None))
    ])
    
    # 2. Nominal: Constant Imputation ('Missing') -> OneHotEncoding (no drop)
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Ordinal: Constant Imputation (median) -> StepEncoder
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('step_encoder', StepEncoder())
    ])
    
    # Combine them using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', make_passthrough_transformer(), binary_cols),
            ('nominal', nominal_transformer, nominal_cols),
            ('ordinal', ordinal_transformer, ordinal_cols),
            ('continuous', continuous_transformer, continuous_cols)
        ],
        remainder='drop'  # Drop anything else (like target or ID columns)
    )
    
    return preprocessor
