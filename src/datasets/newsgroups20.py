from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from .base import BaseDataset


class Newsgroups20Dataset(BaseDataset):
    path = BaseDataset.data_dir / "newsgroups20.csv"

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.lower().split())

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["text"] = out["text"].astype("string").map(self._clean_text)
        return out

    def _download_split(self, split: str) -> pd.DataFrame:
        dataset = fetch_20newsgroups(
            subset=split,
            remove=("headers", "footers", "quotes"),
        )
        return pd.DataFrame(
            {
                "split": split,
                "text": dataset.data,
                "target": dataset.target,
            }
        )

    def load(self, force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        if force_download or not self.path.exists():
            self.ensure_data_dir()
            df = pd.concat([self._download_split("train"), self._download_split("test")], ignore_index=True)
            df = self.preprocess(df)
            df.to_csv(self.path, index=False)

        df = pd.read_csv(self.path)
        train = df[df["split"] == "train"].reset_index(drop=True)
        test = df[df["split"] == "test"].reset_index(drop=True)
        return train, test


newsgroups20_dataset = Newsgroups20Dataset()


def _clean_text(text: str) -> str:
    return newsgroups20_dataset._clean_text(text)


def _download_split(split: str) -> pd.DataFrame:
    return newsgroups20_dataset._download_split(split)


def _load(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    return newsgroups20_dataset.load(force_download=force_download)
