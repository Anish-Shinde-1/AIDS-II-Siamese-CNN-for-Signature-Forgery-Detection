"""
Offline Handwritten Signature Verification package.
Provides dataset loaders, Siamese CNN architectures, training loops,
evaluation metrics (EER, FAR, FRR, ROC-AUC), and inference verification utilities.
"""

from .model import SignatureFeatureExtractor, SiameseNetwork, count_parameters
from .dataset import SignaturePairDataset, get_dataloaders
from .verify import verify_signatures, load_verification_model, preprocess_signature

__all__ = [
    "SignatureFeatureExtractor",
    "SiameseNetwork",
    "count_parameters",
    "SignaturePairDataset",
    "get_dataloaders",
    "verify_signatures",
    "load_verification_model",
    "preprocess_signature",
]
