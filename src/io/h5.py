import h5py
from pathlib import Path

class H5Writer:
    def __init__(self, path: Path, mode: str):
        self.path = path
        self.mode = mode
        self.file: h5py.File | None = None

    def __enter__(self):
        self.file = h5py.File(self.path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, key: str, data: dict):
        """
            data = {
                "keypoints": ...,
                ...
                "descriptors": ...,
            }
        """
        if self.file is None:
            raise RuntimeError("File is not opened. Use context manager")

        group = self.file.require_group(key)
        for name, value in data.items():
            if name in group:
                del group[name]
            group.create_dataset(name, data=value)

    def write_batch(self, batch: dict):
        """
            batch = {
                "img1": {"keypoints": ..., "descriptors": ...},
                "img2": {...}
            }
        """
        if self.file is None:
            raise RuntimeError("File is not opened. Use context manager")

        for key, data in batch.items():
            self.write(key, data)

        self.file.flush()


class H5Reader:
    def __init__(self, path: Path):
        self.path = path
        self.file: h5py.File | None = None

    def __enter__(self):
        self.file = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            self.file = None

    def read(self, key: str) -> dict:
        """
        Возвращает данные одной группы:

        {
            "keypoints": np.ndarray,
            "descriptors": np.ndarray,
            ...
        }
        """
        if self.file is None:
            raise RuntimeError("File is not opened. Use context manager")

        if key not in self.file:
            raise KeyError(f"Key '{key}' not found in H5 file")

        group = self.file[key]
        return {name: group[name][()] for name in group.keys()}

    def read_batch(self, keys: list[str] | None = None) -> dict:
        """
        {
            "img1": {...},
            "img2": {...}
        }
        """
        if self.file is None:
            raise RuntimeError("File is not opened. Use context manager")

        if keys is None:
            keys = list(self.file.keys())

        return {key: self.read(key) for key in keys}

    def keys(self) -> list[str]:
        if self.file is None:
            raise RuntimeError("File is not opened. Use context manager")
        return list(self.file.keys())
