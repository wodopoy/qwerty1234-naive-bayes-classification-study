from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import MultinomialNB

from .base import BaseClassifier


class MultinomialNBClassifier(BaseClassifier):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self) -> MultinomialNB:
        return MultinomialNB(**self.params)
