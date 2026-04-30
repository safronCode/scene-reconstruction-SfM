from dataclasses import dataclass, field
from typing import Self

from .base import BaseExtractorConfig, PreprocessingConfig


@dataclass
class SuperPointConfiguration:
    descriptor_dim: int
    max_num_keypoints: int
    detection_threshold: float
    nms_radius: int
    remove_borders: int

@dataclass(frozen=True, slots=True)
class SuperPointExtractorConfig(BaseExtractorConfig[SuperPointConfiguration]):
    name: str = "superpoint"
    output_type: str = "feats-superpoint"
    configuration: SuperPointConfiguration = field(default_factory=SuperPointConfiguration)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def create_by_dict(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            output_type=data["output"],
            configuration=SuperPointConfiguration(**data["configuration"]),
            preprocessing=PreprocessingConfig(**data["preprocessing"]),
        )
