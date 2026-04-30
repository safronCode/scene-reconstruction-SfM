from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.constants import BASE_DIR
from .general import GeneralConfig
from .extractor import EXTRACTOR_CONFIGS
from .matcher import MATCHER_CONFIGS

if TYPE_CHECKING:
    from pathlib import Path
    from .extractor import BaseExtractorConfig
    from .matcher import BaseMatcherConfig


@dataclass(frozen=True, slots=True)
class Settings:
    general: GeneralConfig
    extractor: BaseExtractorConfig
    matcher: BaseMatcherConfig


    @classmethod
    def from_yaml(
            cls,
            yaml_path: Path = BASE_DIR / "configs" / "pipeline.yaml",
            extractor_conf_path: Path | None = None,
            matcher_conf_path: Path | None = None,
    ) -> "Settings":
        pipeline_conf: dict = cls._load_yaml(yaml_path)

        extractor_name: str = pipeline_conf["extractor"]
        extractor_conf_path: Path = extractor_conf_path or BASE_DIR / "configs" / "extractors" / f'{extractor_name}.yaml'

        matcher_name: str = pipeline_conf["matcher"]
        matcher_conf_path: Path = matcher_conf_path or BASE_DIR / "configs" / "matchers" / f'{matcher_name}.yaml'

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
    def _load_yaml(path: Path) -> dict:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
