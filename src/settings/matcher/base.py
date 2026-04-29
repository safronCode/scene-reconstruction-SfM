from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Self

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BaseMatcherConfig(Generic[T], ABC):
    name: str
    output_type: str
    configuration: T

    @classmethod
    @abstractmethod
    def create_by_dict(cls, data: dict) -> Self:
        raise NotImplementedError
