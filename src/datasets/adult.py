from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

from .base import BaseDataset


class AdultDataset(BaseDataset):
    x_path = BaseDataset.data_dir / "adult_X.csv"
    y_path = BaseDataset.data_dir / "adult_y.csv"

    def preprocess(self, features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = features.copy()
        y = target.iloc[:, 0].copy()

        object_columns = X.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            X[column] = X[column].astype("string").str.strip()

        X = X.replace("?", pd.NA)
        valid_rows = ~X.isna().any(axis=1)
        X = X.loc[valid_rows].reset_index(drop=True)
        y = y.loc[valid_rows].reset_index(drop=True)

        y = y.astype("string").str.strip()
        return X, y

    def load(self, force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        if force_download or not (self.x_path.exists() and self.y_path.exists()):
            self.ensure_data_dir()
            adult = fetch_ucirepo(id=2)
            X, y = self.preprocess(adult.data.features, adult.data.targets)
            X.to_csv(self.x_path, index=False)
            y.to_frame(name="target").to_csv(self.y_path, index=False)

        X = pd.read_csv(self.x_path)
        y = pd.read_csv(self.y_path)["target"]
        return X, y


adult_dataset = AdultDataset()


def _preprocess(features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return adult_dataset.preprocess(features, target)


def _load(force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    return adult_dataset.load(force_download=force_download)
