from __future__ import annotations

from typing import Any

from sklearn.svm import SVC

from .base import BaseClassifier


class SVCClassifier(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        if "probability" in params and params["probability"] is False:
            raise ValueError("SVCClassifier requires probability=True for evaluate metrics")

        defaults = {"probability": True}
        defaults.update(params)
        self.params = defaults
        self.model = self._build_model()

    def _build_model(self) -> SVC:
        return SVC(**self.params)
