from dataclasses import dataclass, field
from typing import Self

from .base import BaseExtractorConfig, PreprocessingConfig


@dataclass
class DISKConfiguration:
    descriptor_dim: int
    max_num_keypoints: int
    detection_threshold: float
    nms_radius: int
    weights: str
    pad_if_not_divisible: bool

@dataclass(frozen=True, slots=True)
class DISKExtractorConfig(BaseExtractorConfig[DISKConfiguration]):
    name: str = "disk"
    output_type: str = "feats-disk"
    configuration: DISKConfiguration = field(default_factory=DISKConfiguration)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def create_by_dict(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            output_type=data["output"],
            configuration=DISKConfiguration(**data["configuration"]),
            preprocessing=PreprocessingConfig(**data["preprocessing"]),
        )
