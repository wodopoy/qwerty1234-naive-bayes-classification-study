from __future__ import annotations

from io import BytesIO
from typing import Any, Literal
from zipfile import ZipFile

import pandas as pd
import requests
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .base import BaseDataset

FeatureMode = Literal["bow", "tfidf", "hybrid_tfidf"]


class SmsSpamCollectionDataset(BaseDataset):
    dataset_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    csv_path = BaseDataset.data_dir / "sms_spam_collection.csv"

    def load(self, force_download: bool = False) -> pd.DataFrame:
        if force_download or not self.csv_path.exists():
            self.ensure_data_dir()
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as zf, zf.open("SMSSpamCollection") as dataset_file:
                df = pd.read_csv(
                    dataset_file,
                    sep="\t",
                    header=None,
                    names=["label", "text"],
                )
            df.to_csv(self.csv_path, index=False)

        return pd.read_csv(self.csv_path)

    def split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return super().split(
            df,
            label_column="label",
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )

    def preprocess(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mode: FeatureMode = "tfidf",
        max_features: int = 20000,
    ) -> dict[str, Any]:
        def _clean(text: str) -> str:
            return " ".join(str(text).lower().split())

        X_train_text = train_df["text"].map(_clean)
        X_val_text = val_df["text"].map(_clean)
        X_test_text = test_df["text"].map(_clean)

        y_train = (train_df["label"] == "spam").astype(int).to_numpy()
        y_val = (val_df["label"] == "spam").astype(int).to_numpy()
        y_test = (test_df["label"] == "spam").astype(int).to_numpy()

        if mode == "bow":
            vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=max_features)
            X_train = vectorizer.fit_transform(X_train_text)
            X_val = vectorizer.transform(X_val_text)
            X_test = vectorizer.transform(X_test_text)
            artifacts: dict[str, Any] = {"vectorizer": vectorizer}
        elif mode == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=max_features,
                sublinear_tf=True,
            )
            X_train = vectorizer.fit_transform(X_train_text)
            X_val = vectorizer.transform(X_val_text)
            X_test = vectorizer.transform(X_test_text)
            artifacts = {"vectorizer": vectorizer}
        elif mode == "hybrid_tfidf":
            word_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=max_features,
                sublinear_tf=True,
            )
            char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=max_features // 2,
                sublinear_tf=True,
            )

            X_train_word = word_vectorizer.fit_transform(X_train_text)
            X_val_word = word_vectorizer.transform(X_val_text)
            X_test_word = word_vectorizer.transform(X_test_text)

            X_train_char = char_vectorizer.fit_transform(X_train_text)
            X_val_char = char_vectorizer.transform(X_val_text)
            X_test_char = char_vectorizer.transform(X_test_text)

            X_train = hstack([X_train_word, X_train_char], format="csr")
            X_val = hstack([X_val_word, X_val_char], format="csr")
            X_test = hstack([X_test_word, X_test_char], format="csr")

            artifacts = {
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "artifacts": artifacts,
            "mode": mode,
        }


sms_spam_collection_dataset = SmsSpamCollectionDataset()


def _load(force_download: bool = False) -> pd.DataFrame:
    return sms_spam_collection_dataset.load(force_download=force_download)


def _split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return sms_spam_collection_dataset.split(
        df,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )


def _preprocess(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: FeatureMode = "tfidf",
    max_features: int = 20000,
) -> dict[str, Any]:
    return sms_spam_collection_dataset.preprocess(
        train_df,
        val_df,
        test_df,
        mode=mode,
        max_features=max_features,
    )
