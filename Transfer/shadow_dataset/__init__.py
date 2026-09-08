"""Shadow dataset preparation and loading pipeline."""

from shadow_dataset.constants import SCHEMA_VERSION
from shadow_dataset.dataset import ShadowCropDataset
from shadow_dataset.letterbox import letterbox_semantic_guide

__all__ = [
    "SCHEMA_VERSION",
    "ShadowCropDataset",
    "letterbox_semantic_guide",
]

__version__ = "0.1.0"
