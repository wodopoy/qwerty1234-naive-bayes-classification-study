from __future__ import annotations

from typing import Any

from sklearn.tree import DecisionTreeClassifier

from .base import BaseClassifier


class DecisionTreeClassifierModel(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**self.params)
