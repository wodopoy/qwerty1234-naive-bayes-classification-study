from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):
    """Base contract for all datasets in the project."""

    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    @abstractmethod
    def load(self, force_download: bool = False) -> Any:
        """Load dataset from local cache or download source data."""
        raise NotImplementedError

    def split(
        self,
        df: pd.DataFrame,
        label_column: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if val_size + test_size >= 1:
            raise ValueError("val_size + test_size must be < 1")

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[label_column],
            random_state=random_state,
        )
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df[label_column],
            random_state=random_state,
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    @abstractmethod
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        """
        Apply dataset-specific preprocessing and return model-ready objects.

        Contract:
        - Input: raw dataset parts from `load()` or `split()` (depends on dataset).
        - Output must be one of:
          1) tuple[pd.DataFrame, pd.Series] for tabular X/y datasets, or
          2) dict with train/val/test features and targets for text pipelines.
        - Output must be deterministic for the same input and params.
        """
        raise NotImplementedError

    @classmethod
    def ensure_data_dir(cls) -> None:
        cls.data_dir.mkdir(parents=True, exist_ok=True)
