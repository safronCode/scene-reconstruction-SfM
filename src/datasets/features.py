from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureImageStats:
    image_name: str
    keypoints_count: int


@dataclass(frozen=True)
class FeatureSummary:
    image_count: int
    min_keypoints: int
    mean_keypoints: float
    max_keypoints: int
    images: list[FeatureImageStats]
