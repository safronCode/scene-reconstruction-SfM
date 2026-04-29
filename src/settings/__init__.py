from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .general import GeneralConfig

from .extractor import EXTRACTOR_CONFIGS
from .extractor.base import BaseExtractorConfig

from .matcher import MATCHER_CONFIGS
from .matcher.base import BaseMatcherConfig

BASE_DIR = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    general: GeneralConfig
    extractor: BaseExtractorConfig
    matcher: BaseMatcherConfig


    @classmethod
    def from_yaml(
            cls,
            yaml_path: Path | str = BASE_DIR / "configs" / "pipeline.yaml",
            extractor_conf_path: Any = None,
            matcher_conf_path: Any = None,

    ) -> "Settings":
        pipeline_conf: dict = cls._load_yaml(yaml_path)

        extractor_name: str = pipeline_conf["extractor"]
        extractor_conf_path: str | Path = extractor_conf_path or BASE_DIR / "configs" / "extractors" / f'{extractor_name}.yaml'

        matcher_name: str = pipeline_conf["matcher"]
        matcher_conf_path: str | Path = extractor_conf_path or BASE_DIR / "configs" / "matchers" / f'{matcher_name}.yaml'

        return cls(
            general=GeneralConfig.create_by_dict(pipeline_conf["general"]),
            extractor=EXTRACTOR_CONFIGS[extractor_name].create_by_dict(
                cls._load_yaml(extractor_conf_path)
            ),
            matcher=MATCHER_CONFIGS[matcher_name].create_by_dict(
                cls._load_yaml(matcher_conf_path)
            )
        )


    @staticmethod
    def _load_yaml(path: Path | str) -> dict:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)