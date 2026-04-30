import cv2
import torch
import numpy as np

from src.core.config.extractor.base import PreprocessingConfig


def resize_image(image:np.ndarray, size:tuple) -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = size[::-1]

    if target_w < w or target_h < h:
        interpolation = cv2.INTER_AREA # Сжатие
    else:
        interpolation = cv2.INTER_LINEAR # Расширение

    resized_image = cv2.resize(image, size, interpolation=interpolation)
    return resized_image



class ImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_path,
            configuration: PreprocessingConfig,
    ):
        self.root_path  = root_path
        self.conf = configuration

        self.image_paths = sorted([
            file for file in list(root_path.iterdir())
            if file.is_file() and file.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ])

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        name = self.image_paths[idx]
        image = self._read_image(name)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]  # WxH

        if self.conf.resize_max and (
                self.conf.resize_force or max(size) > self.conf.resize_max
        ):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new)

        image = image[None]
        image = image / 255.0 # Scale от 0 до 1

        data = {
            "path": str(name),
            "image": image,
            "original_size": np.array(size),
        }
        return data


    def __len__(self):
        return len(self.image_paths)

    def _read_image(self, path):
        grayscale = self.conf.grayscale

        if grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR

        image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)

        if len(image.shape) == 3:
            image = image[:, :, ::-1]  #BGR --> RGB

        return image

    def get_all_pairs(self) -> list[tuple[str, str]]:
        from itertools import combinations

        names = [file.name for file in self.image_paths]
        return list(combinations(names, 2))
