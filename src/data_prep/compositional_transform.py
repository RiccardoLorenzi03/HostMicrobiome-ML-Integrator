"""Compositional data transformations for microbiome features."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Apply Centered Log-Ratio (CLR) transformation to compositional data."""

    def __init__(self, pseudocount: float = 1e-6) -> None:
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        """No-op fit to follow scikit-learn's transformer API."""
        return self

    def transform(self, X):
        """Return CLR-transformed microbiome compositional features."""
        X_arr = np.asarray(X, dtype=float)
        if np.any(X_arr < 0):
            raise ValueError("CLR transformation requires non-negative compositional values.")

        adjusted = X_arr + self.pseudocount
        geometric_mean = np.exp(np.mean(np.log(adjusted), axis=1, keepdims=True))
        return np.log(adjusted / geometric_mean)
