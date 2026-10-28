import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, model_validator

from hubai_sdk.typing import License, Task
from hubai_sdk.utils.hubai_models import PerformanceRating, PerformanceRatings


class ArchitectureConfig(BaseModel):
    name: str
    description: str = ""
    license_type: License = "undefined"
    is_public: bool | None = False
    tasks: list[Task] | None = None
    training_image: str = "luxonis-train"
    training_model_config: dict[str, Any] | None = None
    training_loader_config: dict[str, Any] | None = None
    training_hyperparameter_config: dict[str, Any] | None = None
    performance_ratings: PerformanceRatings | dict[str, Any] | None = None
    config_file_path: str | Path | None = None

    @model_validator(mode="before")
    @classmethod
    def load_config_from_file(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load training configs from file if config_file_path is
        provided."""
        config_file_path = data.get("config_file_path")
        training_model_config = data.get("training_model_config")
        training_loader_config = data.get("training_loader_config")
        training_hyperparameter_config = data.get(
            "training_hyperparameter_config"
        )

        # Check mutual exclusivity
        if config_file_path is not None and (
            training_model_config is not None
            or training_loader_config is not None
            or training_hyperparameter_config is not None
        ):
            raise ValueError(
                "Cannot provide both config_file_path and "
                "training_model_config/training_loader_config/"
                "training_hyperparameter_config"
            )

        # Check that at least one option is provided
        if config_file_path is None and (
            training_model_config is None
            or training_hyperparameter_config is None
        ):
            raise ValueError(
                "Must provide either config_file_path or both "
                "training_model_config and training_hyperparameter_config"
            )

        # Load from file if provided
        if config_file_path is not None:
            file_path = Path(config_file_path)

            if not file_path.exists():
                raise ValueError(
                    f"Config file '{config_file_path}' does not exist."
                )

            # Load based on file extension
            if file_path.suffix == ".json":
                with open(file_path) as f:
                    config = json.load(f)
                logger.info(f"Loaded JSON config from '{config_file_path}'")
            elif file_path.suffix in [".yml", ".yaml"]:
                with open(file_path) as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded YAML config from '{config_file_path}'")
            else:
                raise ValueError(
                    f"Config file '{config_file_path}' must be JSON or YAML"
                )

            # Extract configs
            data["training_model_config"] = config.get("model")
            data["training_loader_config"] = config.get("loader")
            data["training_hyperparameter_config"] = config.get("trainer")

            if data["training_model_config"] is None:
                raise ValueError(
                    f"Config file '{config_file_path}' does not contain 'model' key"
                )
            if data["training_hyperparameter_config"] is None:
                raise ValueError(
                    f"Config file '{config_file_path}' does not contain 'trainer' key"
                )

        # Remove config_file_path from data as it's not stored
        data.pop("config_file_path", None)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_performance_ratings(
        cls, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Set default performance ratings if not provided."""
        if data.get("performance_ratings") is None:
            data["performance_ratings"] = PerformanceRatings(
                rvc2_ratings=PerformanceRating(
                    accuracy_rating=None, speed_rating=None, fps_rating=None
                ),
                rvc3_ratings=PerformanceRating(
                    accuracy_rating=None, speed_rating=None, fps_rating=None
                ),
                rvc4_ratings=PerformanceRating(
                    accuracy_rating=None, speed_rating=None, fps_rating=None
                ),
                hailo_ratings=PerformanceRating(
                    accuracy_rating=None, speed_rating=None, fps_rating=None
                ),
            )
        return data
