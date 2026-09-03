import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class StepEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to step-encode (cumulative encode) ordinal features.
    For a feature with unique values [1, 2, 3], it creates binary columns:
    Feature_>=2, Feature_>=3.
    """
    def __init__(self):
        self.feature_values_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_arr = X.values
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)
            
        for i, col_name in enumerate(self.feature_names_in_):
            # Find unique sorted values, ignoring NaNs
            unique_vals = np.unique(X_arr[:, i])
            unique_vals = unique_vals[~pd.isna(unique_vals)]
            unique_vals = np.sort(unique_vals)
            # We don't need a threshold for the minimum value because it would just be all 1s
            if len(unique_vals) > 1:
                self.feature_values_[col_name] = unique_vals[1:]
            else:
                self.feature_values_[col_name] = []
                
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            
        output_cols = []
        out_data = []
        
        for i, col_name in enumerate(self.feature_names_in_):
            col_data = X_arr[:, i]
            thresholds = self.feature_values_.get(col_name, [])
            for thresh in thresholds:
                # 1 if >= thresh, 0 otherwise. NaNs are propagated as 0 or handled before.
                # Assuming imputation handles NaNs before this step.
                binary_col = (col_data >= thresh).astype(int)
                out_data.append(binary_col)
                # Ensure integer printing for float thresholds if they are round
                thresh_str = int(thresh) if thresh == int(thresh) else thresh
                output_cols.append(f"{col_name}_>={thresh_str}")
                
        if not out_data:
            empty_arr = np.empty((X.shape[0], 0))
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(empty_arr, index=X.index)
            return empty_arr
            
        transformed_X = np.column_stack(out_data)
        
        # Scikit-learn API expects numpy array returned.
        # We also attach column names for downstream retrieval.
        self._output_columns = output_cols
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed_X, columns=output_cols, index=X.index)
        return transformed_X

    def get_feature_names_out(self, input_features=None):
        return np.array(self._output_columns)
