from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier

from .base import BaseClassifier


class KNeighborsClassifierModel(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(**self.params)
