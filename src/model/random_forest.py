from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier


class RandomForestClassifierModel(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.params)
