from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING

from src.io.h5 import H5Writer
from src.utils import get_best_device

from src.features import extractors
from src.utils.dynamic_load import dynamic_load_model

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path
    from src.datasets import ImageDataset
    from src.core.config.extractor import BaseExtractorConfig


def _extract_one(
        data,
        model,
        device: torch.device,
) -> dict:
    size = np.array(data["image"].shape[-2:][::-1])
    original_size = data["original_size"][0].numpy()
    scales = (original_size / size).astype(np.float32)

    prediction = model({"image": data["image"].to(device, non_blocking=True)})
    prediction = {key: value[0].cpu().numpy() for key, value in prediction.items()}

    prediction["image_size"] = original_size
    if "keypoints" in prediction:
        prediction["keypoints"] = (prediction["keypoints"] + 0.5) * scales[None] - 0.5
        prediction["keypoint_uncertainty"] = np.array(
            [getattr(model, "detection_noise", 1) * scales.mean()],
            dtype=np.float32,
        )

    if "scales" in prediction:
        prediction["scales"] *= scales.mean()
    return prediction


def extract_features(
        extractor_conf: BaseExtractorConfig,
        images: ImageDataset,
        save_h5_path: Path,
        device: torch.device = None,
) -> Path:
    logger.info("Extracting features with %s conf for %d images\n"
                "conf: %s", extractor_conf.name, len(images), extractor_conf.configuration)

    if device is None:
        device = get_best_device()

    extractor_class = dynamic_load_model(extractors, extractor_conf.name)
    extractor = extractor_class(extractor_conf.configuration).eval().to(device)

    loader = torch.utils.data.DataLoader(
        images,
        pin_memory=device.type == "cuda"
    )

    features_path = save_h5_path / f"features_{extractor_conf.name}.h5"
    with H5Writer(path=features_path, mode="w") as h5_writer:
        with torch.inference_mode():
            for idx, data in enumerate(tqdm(loader)):
                image_name: str = images.image_paths[idx].name
                predicted_features = _extract_one(data, extractor, device)

                h5_writer.write(
                    key=image_name,
                    data=predicted_features
                )

    logger.info(f"Finished extracting features. Wrote results to {features_path}")
    return features_path
