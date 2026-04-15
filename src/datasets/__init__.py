from .adult import AdultDataset, _load as load_adult
from .base import BaseDataset
from .breast_cancer_wisconsin import (
    BreastCancerWisconsinDataset,
    _load as load_breast_cancer_wisconsin,
)
from .digits import DigitsDataset, _load as load_digits
from .newsgroups20 import Newsgroups20Dataset, _load as load_newsgroups20
from .sms_spam_collection import SmsSpamCollectionDataset, _load as load_sms_spam_collection
from .spambase import SpambaseDataset, _load as load_spambase

__all__ = [
    "BaseDataset",
    "AdultDataset",
    "BreastCancerWisconsinDataset",
    "DigitsDataset",
    "Newsgroups20Dataset",
    "SmsSpamCollectionDataset",
    "SpambaseDataset",
    "load_adult",
    "load_breast_cancer_wisconsin",
    "load_digits",
    "load_newsgroups20",
    "load_sms_spam_collection",
    "load_spambase",
]
