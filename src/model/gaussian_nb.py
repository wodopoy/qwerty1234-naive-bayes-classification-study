from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.naive_bayes import GaussianNB

from .base import BaseClassifier


class GaussianNBClassifier(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> GaussianNB:
        return GaussianNB(**self.params)

    def fit(self, X: Any, y: Any) -> "GaussianNBClassifier":
        X_dense = self._to_dense_if_sparse(X)
        self.model.fit(X_dense, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._require_fitted()
        X_dense = self._to_dense_if_sparse(X)
        return np.asarray(self.model.predict(X_dense))

    def predict_proba(self, X: Any) -> np.ndarray:
        self._require_fitted()
        X_dense = self._to_dense_if_sparse(X)
        return np.asarray(self.model.predict_proba(X_dense))
