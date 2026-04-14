from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import pandas as pd
import requests
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
CSV_PATH = DATA_DIR / "sms_spam_collection.csv"
FeatureMode = Literal["bow", "tfidf", "hybrid_tfidf"]


def _load(force_download: bool = False) -> pd.DataFrame:
    if force_download or not CSV_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as zf, zf.open("SMSSpamCollection") as dataset_file:
            df = pd.read_csv(
                dataset_file,
                sep="\t",
                header=None,
                names=["label", "text"],
            )
        df.to_csv(CSV_PATH, index=False)

    return pd.read_csv(CSV_PATH)


def _split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _preprocess(
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
