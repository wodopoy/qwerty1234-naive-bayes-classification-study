from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
CSV_PATH = DATA_DIR / "sms_spam_collection.csv"


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
