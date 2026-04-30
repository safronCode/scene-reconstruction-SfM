from pathlib import Path
from dataclasses import dataclass

from src.core import BASE_DIR


@dataclass(frozen=True)
class GeneralConfig:
    """
        Конфигурация базовых настроек:
            object_name [str] - название объекта исследования
            input_dir_path  [Path] - <корень> до директории ввода
            output_dir_path [Path] - <корень> до директории вывода
            * {input_dir_path}/{object_name} - абсолютный путь до директории с входными изображениями
            * {output_dir_path}/{object_name} - абсолютный путь до директории с выходными данными

            logging     [bool] - флаг логирования
            show_calcs  [bool] - флаг демонстрации вычисленных значений
            show_visualizations [bool] - флаг демонстрации визуализаций
    """
    object_name: str
    input_dir_path: Path | str
    output_dir_path: Path | str

    logging: bool = False
    show_calcs: bool = False
    show_visualizations: bool = False

    def __post_init__(self):
        if not self.object_name or not isinstance(self.object_name, str):
            raise ValueError("object_name must be a non-empty string")

        input_dir_path = self.input_dir_path or "default"
        output_dir_path = self.output_dir_path or "default"

        if input_dir_path == "default":
            input_dir_path: Path = BASE_DIR / "data_i" / self.object_name
        else:
            input_dir_path: Path = Path(input_dir_path)

        if output_dir_path == "default":
            output_dir_path: Path = BASE_DIR / "data_o" / self.object_name
        else:
            output_dir_path: Path = Path(output_dir_path)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "input_dir_path", input_dir_path)
        object.__setattr__(self, "output_dir_path", output_dir_path)

    @classmethod
    def create_by_dict(cls, data: dict) -> "GeneralConfig":
        return cls(
            object_name=data.get("object_name"),
            input_dir_path=data.get("input_dir_path", "default"),
            output_dir_path=data.get("output_dir_path", "default"),

            logging=data.get("logging", False),
            show_calcs=data.get("show_calcs", False),
            show_visualizations=data.get("show_visualizations", False),
        )
