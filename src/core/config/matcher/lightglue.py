from dataclasses import dataclass, field
from typing import Self

from .base import BaseMatcherConfig


@dataclass(frozen=True, slots=True)
class LightGlueConfiguration:
    features: str
    filter_threshold: float

@dataclass(frozen=True, slots=True)
class LightGlueMatcherConfig(BaseMatcherConfig[LightGlueConfiguration]):
    name: str
    output_type: str
    configuration: LightGlueConfiguration = field(default_factory=LightGlueConfiguration)

    @classmethod
    def create_by_dict(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            output_type=data["output"],
            configuration=LightGlueConfiguration(**data["configuration"]),
        )
