from __future__ import annotations
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.base import BaseEstimator, TransformerMixin

class CLRTransformer(BaseEstimator, TransformerMixin):
    """Apply Centered Log-Ratio (CLR) transformation to compositional data."""
    def __init__(self, pseudocount: float = 1e-6) -> None:
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if np.any(X_arr < 0):
            raise ValueError("CLR transformation requires non-negative compositional values.")
        adjusted = X_arr + self.pseudocount
        geometric_mean = np.exp(np.mean(np.log(adjusted), axis=1, keepdims=True))
        return np.log(adjusted / geometric_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applica CLR a una matrice composizionale.")
    parser.add_argument("--input", type=str, required=True, help="Path CSV di input.")
    parser.add_argument("--output", type=str, required=True, help="Path CSV di output.")
    args = parser.parse_args()

    print(f"Caricamento dati da {args.input}...")
    df = pd.read_csv(args.input, index_col=0)
    
    transformer = CLRTransformer()
    df_clr = pd.DataFrame(
        transformer.transform(df.values), 
        index=df.index, 
        columns=df.columns
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clr.to_csv(args.output)
    print(f"Dati CLR salvati in {args.output}")
