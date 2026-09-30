import argparse
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio (CLR) transformer for compositional data."""
    def __init__(self, pseudocount: float = 1e-6) -> None:
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if np.any(X_arr < 0):
            raise ValueError("CLR transformation requires non-negative values.")
        adjusted = X_arr + self.pseudocount
        gmean = np.exp(np.mean(np.log(adjusted), axis=1, keepdims=True))
        return np.log(adjusted / gmean)

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CLR transformation.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    df_clr = pd.DataFrame(CLRTransformer().transform(df.values), index=df.index, columns=df.columns)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clr.to_csv(args.output)
    print(f"[INFO] CLR transformed data saved to {args.output}")

if __name__ == "__main__":
    main()
