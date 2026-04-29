from dataclasses import dataclass
from pathlib import Path


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
    input_dir_path: Path | None = None
    output_dir_path: Path | None = None

    logging: bool = False
    show_calcs: bool = False
    show_visualizations: bool = False

    def __post_init__(self):
        if not self.object_name or not isinstance(self.object_name, str):
            raise ValueError("object_name must be a non-empty string")

        # todo доработать - validation из yaml приходит "default"
        if self.input_dir_path is None:
            object.__setattr__(self, "input_dir_path", Path(__file__).parent)

        if self.output_dir_path is None:
            object.__setattr__(self, "output_dir_path", Path(__file__).parent)


    @classmethod
    def create_by_dict(cls, data: dict) -> "GeneralConfig":
        return cls(
            object_name=data.get("object_name"),
            input_dir_path=data.get("input_dir_path"),
            output_dir_path=data.get("output_dir_path"),

            logging=data.get("logging", False),
            show_calcs=data.get("show_calcs", False),
            show_visualizations=data.get("show_visualizations", False),
        )
