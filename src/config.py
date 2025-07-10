from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    data_root: str
    physio_signals_dir: str
    av_ratings_file: str
    sampling_rates: Dict[str, int]
    modalities: list[str]


@dataclass
class TimesFMConfig:
    model_id: str
    context_length: int
    horizon_length: int
    per_core_batch_size: int
    feature_dim: int
    use_gpu: bool
    backend: str
    total_layers: int
    capture_layers: bool


@dataclass
class ExperimentConfig:
    results_dir: str
    max_recordings: Optional[int]
    n_folds: int
    random_state: int
    shuffle: bool
    layer_wise: bool
    layers_to_probe: Optional[list[int]]
    targets: list[str]
    quick_test: Dict[str, Any]


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]


@dataclass
class PreprocessingConfig:
    normalize: bool
    resample_length: Optional[int]
    feature_aggregation: str


@dataclass
class PathsConfig:
    src_dir: str
    data_dir: str
    results_dir: str
    cache_dir: str


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: Optional[str]


@dataclass
class Config:
    dataset: DatasetConfig
    timesfm: TimesFMConfig
    experiment: ExperimentConfig
    models: Dict[str, ModelConfig]
    preprocessing: PreprocessingConfig
    paths: PathsConfig
    logging: LoggingConfig


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Create structured config objects
    dataset = DatasetConfig(**raw_config["dataset"])
    timesfm = TimesFMConfig(**raw_config["timesfm"])
    experiment = ExperimentConfig(**raw_config["experiment"])

    models = {
        name: ModelConfig(**model_data)
        for name, model_data in raw_config["models"].items()
    }

    preprocessing = PreprocessingConfig(**raw_config["preprocessing"])
    paths = PathsConfig(**raw_config["paths"])
    logging_config = LoggingConfig(**raw_config["logging"])

    return Config(
        dataset=dataset,
        timesfm=timesfm,
        experiment=experiment,
        models=models,
        preprocessing=preprocessing,
        paths=paths,
        logging=logging_config,
    )


def get_config() -> Config:
    """Get the global configuration instance."""
    return load_config()


# For backward compatibility, also provide a simple dict-based loader
def load_config_dict(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load configuration as a dictionary."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
