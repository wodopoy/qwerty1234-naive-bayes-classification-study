from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import BernoulliNB

from .base import BaseClassifier


class BernoulliNBClassifier(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> BernoulliNB:
        return BernoulliNB(**self.params)
