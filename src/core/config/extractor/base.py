from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Self

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    grayscale: bool
    resize_max: int
    resize_force: bool

@dataclass(frozen=True, slots=True)
class BaseExtractorConfig(Generic[T], ABC):
    name: str
    output_type: str
    configuration: T
    preprocessing: PreprocessingConfig

    @classmethod
    @abstractmethod
    def create_by_dict(cls, data: dict) -> Self:
        raise NotImplementedError
