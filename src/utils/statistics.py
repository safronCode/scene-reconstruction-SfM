import h5py
import numpy as np
from pathlib import Path

from src.io import H5Reader
from src.datasets import (FeatureImageStats, FeatureSummary,
                          MatchPairStats, MatchSummary)


# Features BLOCK
def _count_matches_streaming(dataset: h5py.Dataset, chunk_size: int = 1_000_000) -> int:
    total = 0
    total_rows = dataset.shape[0]
    for start in range(0, total_rows, chunk_size):
        chunk = dataset[start:start + chunk_size]
        total += np.count_nonzero(chunk != -1)
    return int(total)


def summarize_features_h5(features_h5_path: str | Path) -> FeatureSummary:
    images: list[FeatureImageStats] = []
    with H5Reader(features_h5_path) as reader:
        for image_name in reader.keys():
            group = reader.read(image_name)
            if "keypoints" not in group:
                continue
            images.append(
                FeatureImageStats(
                    image_name=image_name,
                    keypoints_count=int(group["keypoints"].shape[0]),
                )
            )

    if not images:
        raise ValueError("No feature groups with keypoints found. Did feature extraction finish?")

    counts = [image_stats.keypoints_count for image_stats in images]
    return FeatureSummary(
        image_count=len(images),
        min_keypoints=min(counts),
        mean_keypoints=sum(counts) / len(counts),
        max_keypoints=max(counts),
        images=images,
    )

def print_features_summary(features_h5_path: str | Path) -> FeatureSummary:
    summary = summarize_features_h5(features_h5_path)
    print(f"Images in H5: {summary.image_count}")
    print(
        "Keypoints per image: "
        f"min={summary.min_keypoints}, mean={summary.mean_keypoints:.1f}, max={summary.max_keypoints}"
    )
    print(f"Filename\t\t| Keypoint number\n" + "-" * 40)
    for image_stats in summary.images:
        print(f"{image_stats.image_name:15} | {image_stats.keypoints_count:5} kp")
    return summary


# Matches BLOCK
def summarize_matches_h5(
        matches_h5_path: str | Path,
        top_k: int = 5,
        chunk_size: int = 1_000_000,
) -> MatchSummary:
    pair_counts: dict[str, int] = {}

    with H5Reader(matches_h5_path) as reader:
        h5_file = reader.file
        if h5_file is None:
            raise RuntimeError("Matches H5 file is not opened.")

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset) and name.endswith("/matches0"):
                pair_name = name.rsplit("/matches0", 1)[0]
                pair_counts[pair_name] = _count_matches_streaming(obj, chunk_size=chunk_size)

        h5_file.visititems(visitor)

    if not pair_counts:
        raise ValueError("No matches0 datasets found. Did matching finish?")

    counts = list(pair_counts.values())
    top_pairs = [
        MatchPairStats(pair_name=pair_name, matches_count=matches_count)
        for pair_name, matches_count in sorted(
            pair_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]

    return MatchSummary(
        pair_count=len(counts),
        min_matches=min(counts),
        mean_matches=sum(counts) / len(counts),
        max_matches=max(counts),
        top_pairs=top_pairs,
    )


def print_matches_summary(
        matches_h5_path: str | Path,
        top_k: int = 5,
        chunk_size: int = 1_000_000,
) -> MatchSummary:
    summary = summarize_matches_h5(
        matches_h5_path=matches_h5_path,
        top_k=top_k,
        chunk_size=chunk_size,
    )
    print(f"Pairs: {summary.pair_count}")
    print(
        "Matches per pair: "
        f"min={summary.min_matches}, mean={summary.mean_matches:.1f}, max={summary.max_matches}"
    )
    print(f"Top-{len(summary.top_pairs)} pairs by matches:")
    for pair_stats in summary.top_pairs:
        print(f"  {pair_stats.pair_name}: {pair_stats.matches_count}")
    return summary
