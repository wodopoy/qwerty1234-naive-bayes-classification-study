from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        defaults = {"max_iter": 1000}
        defaults.update(params)
        self.params = defaults
        self.model = self._build_model()

    def _build_model(self) -> LogisticRegression:
        return LogisticRegression(**self.params)
