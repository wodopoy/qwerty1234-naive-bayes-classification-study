from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.sparse import issparse
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseClassifier(ABC):
    """Minimal contract for classifiers used in this project."""

    def __init__(self) -> None:
        self.model: Any | None = None

    @abstractmethod
    def _build_model(self) -> Any:
        """Return configured sklearn estimator."""

    def fit(self, X: Any, y: Any) -> "BaseClassifier":
        if self.model is None:
            self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        self._require_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.__class__.__name__} does not support predict_proba")
        return np.asarray(self.model.predict_proba(X))

    def evaluate(
        self,
        X: Any,
        y: Any,
        n_bins: int = 10,
        calibration_strategy: str = "uniform",
    ) -> dict[str, Any]:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        classes = np.unique(y_true)
        n_classes = int(classes.size)
        is_binary = n_classes == 2

        if is_binary:
            positive_index = 1 if y_proba.shape[1] > 1 else 0
            pos_proba = y_proba[:, positive_index]
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, pos_proba)),
                "log_loss": float(log_loss(y_true, y_proba, labels=classes)),
            }
            prob_true, prob_pred = calibration_curve(
                y_true,
                pos_proba,
                n_bins=n_bins,
                strategy=calibration_strategy,
            )
            calibration: dict[str, Any] | None = {
                "prob_true": prob_true,
                "prob_pred": prob_pred,
                "n_bins": int(n_bins),
                "strategy": calibration_strategy,
            }
            task_type = "binary"
        else:
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "roc_auc": float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                ),
                "log_loss": float(log_loss(y_true, y_proba, labels=classes)),
            }
            calibration = None
            task_type = "multiclass"

        return {
            "metrics": metrics,
            "calibration": calibration,
            "meta": {
                "task_type": task_type,
                "n_classes": n_classes,
                "model_name": self.__class__.__name__,
            },
        }

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit before predict/evaluate.")

    @staticmethod
    def _to_dense_if_sparse(X: Any) -> Any:
        if issparse(X):
            return X.toarray()
        return X
