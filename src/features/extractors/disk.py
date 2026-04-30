import torch
import kornia

from src.features.base import BaseModel
from src.utils import get_best_device


class DISK(BaseModel):
    required_data_keys = ["image"]

    def _init(self, conf) -> None:
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights, device=get_best_device())

    def _forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.ndim == 5 and image.shape[1] == 1 and image.shape[-1] in (1, 3):
            image = image.squeeze(1).permute(0, 3, 1, 2).contiguous()
        elif image.ndim == 4 and image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2).contiguous()
        elif image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape[1] == 1:
            image = kornia.color.grayscale_to_rgb(image)
        features = self.model(
            image,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_radius,
            score_threshold=self.conf.detection_threshold,
            pad_if_not_divisible=self.conf.pad_if_not_divisible,
        )
        keypoints = [f.keypoints for f in features]
        scores = [f.detection_scores for f in features]
        descriptors = [f.descriptors for f in features]
        del features

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints.to(image).contiguous(),
            "keypoint_scores": scores.to(image).contiguous(),
            "descriptors": descriptors.to(image).contiguous(),
        }
