from .base import BaseExtractorConfig
from .disk import DISKExtractorConfig
from .superpoint import SuperPointExtractorConfig

EXTRACTOR_CONFIGS = {
    "superpoint": SuperPointExtractorConfig,
    "disk": DISKExtractorConfig,
}

