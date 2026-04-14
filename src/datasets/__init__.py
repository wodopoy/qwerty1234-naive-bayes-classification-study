from .adult import AdultDataset, _load as load_adult
from .base import BaseDataset

from .sms_spam_collection import SmsSpamCollectionDataset, _load as load_sms_spam_collection

__all__ = [
    "BaseDataset",
    "SmsSpamCollectionDataset",
    "load_sms_spam_collection",
]
