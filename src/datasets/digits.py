from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_digits

from .base import BaseDataset


class DigitsDataset(BaseDataset):
    x_path = BaseDataset.data_dir / "digits_X.csv"
    y_path = BaseDataset.data_dir / "digits_y.csv"

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        return X, y

    def load(self, force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        if force_download or not (self.x_path.exists() and self.y_path.exists()):
            self.ensure_data_dir()
            digits = load_digits()
            feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
            X = pd.DataFrame(digits.data, columns=feature_names)
            y = pd.Series(digits.target, name="target")
            X, y = self.preprocess(X, y)
            X.to_csv(self.x_path, index=False)
            y.to_frame().to_csv(self.y_path, index=False)

        X = pd.read_csv(self.x_path)
        y = pd.read_csv(self.y_path)["target"]
        return X, y


digits_dataset = DigitsDataset()


def _load(force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    return digits_dataset.load(force_download=force_download)
