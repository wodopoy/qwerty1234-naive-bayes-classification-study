from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

from .base import BaseDataset


class BreastCancerWisconsinDataset(BaseDataset):
    x_path = BaseDataset.data_dir / "breast_cancer_wisconsin_X.csv"
    y_path = BaseDataset.data_dir / "breast_cancer_wisconsin_y.csv"

    def preprocess(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = features.copy()
        y = target.iloc[:, 0].copy()

        X = X.replace("?", pd.NA)
        valid_rows = ~X.isna().any(axis=1)
        X = X.loc[valid_rows].reset_index(drop=True)
        y = y.loc[valid_rows].reset_index(drop=True)

        return X, y

    def load(self, force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        if force_download or not (self.x_path.exists() and self.y_path.exists()):
            self.ensure_data_dir()
            bc = fetch_ucirepo(id=17)
            X, y = self.preprocess(bc.data.features, bc.data.targets)
            X.to_csv(self.x_path, index=False)
            y.to_frame(name="target").to_csv(self.y_path, index=False)

        X = pd.read_csv(self.x_path)
        y = pd.read_csv(self.y_path)["target"]
        return X, y


breast_cancer_wisconsin_dataset = BreastCancerWisconsinDataset()


def _preprocess(features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return breast_cancer_wisconsin_dataset.preprocess(features, target)


def _load(force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    return breast_cancer_wisconsin_dataset.load(force_download=force_download)
