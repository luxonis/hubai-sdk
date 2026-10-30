"""Manage HubAI trainable architecture definitions.

Use ``client.architecture`` or the ``hubai architecture`` CLI group. Create
and update accept either inline model/trainer configuration or a JSON/YAML
file. Deleting an architecture removes it from HubAI.
"""

from typing import Annotated, Any
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from hubai_sdk.errors import InputError
from hubai_sdk.typing import License, Order, Task
from hubai_sdk.utils.architecture_config import ArchitectureConfig
from hubai_sdk.utils.hub import (
    get_resource_info,
    print_hub_ls,
    print_hub_resource_info,
    raise_for_hub_error,
    resolve_resource_id,
    run_cli,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import (
    PerformanceRatings,
)
from hubai_sdk.utils.sdk_models import ArchitectureResponse

app = App(
    name="architecture",
    help="Architectures Interactions",
    group="Resource Management",
)

ARCHITECTURE_LIST_KEYS = ["name", "id", "slug"]
ARCHITECTURE_INFO_KEYS = [
    "name",
    "slug",
    "id",
    "created",
    "updated",
    "tasks",
    "is_public",
    "license_type",
    "description",
    "training_model_config",
    "training_loader_config",
    "training_hyperparameter_config",
]
TrainingModelConfigOption = Annotated[
    dict[str, Any] | None,
    Parameter(
        help=(
            "Inline training model config. Required together with "
            "--training-hyperparameter-config unless --config-file-path is "
            "provided."
        )
    ),
]
TrainingLoaderConfigOption = Annotated[
    dict[str, Any] | None,
    Parameter(help="Optional inline training loader config."),
]
TrainingHyperparameterConfigOption = Annotated[
    dict[str, Any] | None,
    Parameter(
        help=(
            "Inline training hyperparameter config. Required together with "
            "--training-model-config unless --config-file-path is provided."
        )
    ),
]
ConfigFilePathOption = Annotated[
    str | None,
    Parameter(
        help=(
            "Path to a JSON or YAML training config containing 'model' and "
            "'trainer' keys. Required unless both --training-model-config "
            "and --training-hyperparameter-config are provided."
        )
    ),
]


def list_architectures(
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
) -> list[ArchitectureResponse]:
    """List architectures visible to the API key.

    The API does not expose a license filter, so ``license_type`` is applied
    locally while following API cursor pages.
    """
    if license_type is not None:
        architectures: list[ArchitectureResponse] = []
        cursor = None
        while len(architectures) < limit:
            try:
                data, cursor = Request.get_page(
                    service="models",
                    endpoint="architectures",
                    params={
                        "tasks": list(tasks) if tasks else [],
                        "is_public": is_public,
                        "limit": 500,
                        "sort": sort,
                        "order": order,
                        "cursor": cursor,
                    },
                )
            except requests.HTTPError as exc:
                raise_for_hub_error(exc)
            architectures.extend(
                item
                for raw in data
                if (item := ArchitectureResponse(**raw)).license_type.value
                == license_type
            )
            if not cursor:
                break
        return architectures[:limit]

    try:
        data = Request.get(
            service="models",
            endpoint="architectures",
            params={
                "tasks": list(tasks) if tasks else [],
                "is_public": is_public,
                "limit": limit,
                "sort": sort,
                "order": order,
            },
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(exc)

    return [ArchitectureResponse(**item) for item in data]


@app.command(name="ls")
def list_architectures_cli(
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    """List the architectures in HubAI."""
    architectures = run_cli(
        lambda: list_architectures(
            tasks=tasks,
            license_type=license_type,
            is_public=is_public,
            limit=limit,
            sort=sort,
            order=order,
        )
    )
    _print_architecture_list(architectures, field)


def get_architecture(identifier: UUID | str) -> ArchitectureResponse:
    """Get architecture information from HubAI."""
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    data = get_resource_info(identifier, "architectures")

    return ArchitectureResponse(**data)


@app.command(name="info")
def get_architecture_info_cli(identifier: UUID | str) -> None:
    """Get architecture information from HubAI."""
    _print_architecture_info(run_cli(lambda: get_architecture(identifier)))


def create_architecture(
    name: str,
    *,
    description: str | None = None,
    license_type: License = "undefined",
    is_public: bool = False,
    tasks: list[Task] | None = None,
    training_image: str = "luxonis-train",
    training_model_config: dict[str, Any] | None = None,
    training_loader_config: dict[str, Any] | None = None,
    training_hyperparameter_config: dict[str, Any] | None = None,
    performance_ratings: PerformanceRatings | dict[str, Any] | None = None,
    config_file_path: str | None = None,
) -> ArchitectureResponse:
    """Creates a new architecture resource.

    Parameters
    ----------
    name : str
        The name of the architecture.
    license_type : License
        The type of the license.
    is_public : bool
        HubAI currently permits creating private architectures only. The
        server rejects public architectures.
    description : str | None
        Full description of the architecture.
    tasks : list[Task] | None
        List of tasks this architecture supports.
    training_image : str
        The internal training image used in this architecture (DO NOT expose to user). Default is "luxonis-train".
    training_model_config : dict[str, Any] | None
        The `model` configuration block parsed by the training image.
    training_loader_config : dict[str, Any] | None
        The optional `loader` configuration block parsed by the training image.
    training_hyperparameter_config : dict[str, Any] | None
        The `trainer` configuration block parsed by the training image.
    performance_ratings : PerformanceRatings | None
        The speed and accuracy ratings for each HW with this architecture.
    config_file_path : str | None
        Path to a JSON or YAML config file that is used in the luxonis-train for defining the model.
    """

    try:
        architecture_config = ArchitectureConfig(
            name=name,
            description=description or "",
            license_type=license_type,
            is_public=is_public,
            tasks=tasks or [],
            training_image=training_image,
            training_model_config=training_model_config,
            training_loader_config=training_loader_config,
            training_hyperparameter_config=training_hyperparameter_config,
            performance_ratings=performance_ratings,
            config_file_path=config_file_path,
        )
    except PydanticValidationError as exc:
        raise _config_input_error(exc) from exc

    data = architecture_config.model_dump(
        mode="json", exclude={"config_file_path"}
    )

    try:
        res = Request.post(
            service="models", endpoint="architectures", json=data
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc,
            conflict_message=f"Architecture '{name}' already exists",
        )
    logger.info(f"Architecture '{res['name']}' created with ID '{res['id']}'")

    return get_architecture(res["id"])


@app.command(name="create")
def create_architecture_cli(
    name: str,
    *,
    description: str | None = None,
    license_type: License = "undefined",
    is_public: bool = False,
    tasks: list[Task] | None = None,
    training_image: str = "luxonis-train",
    training_model_config: TrainingModelConfigOption = None,
    training_loader_config: TrainingLoaderConfigOption = None,
    training_hyperparameter_config: TrainingHyperparameterConfigOption = None,
    performance_ratings: PerformanceRatings | dict[str, Any] | None = None,
    config_file_path: ConfigFilePathOption = None,
) -> None:
    """Creates a new architecture resource.

    Provide either --config-file-path or both --training-model-config and
    --training-hyperparameter-config.
    """
    architecture = run_cli(
        lambda: create_architecture(
            name,
            description=description,
            license_type=license_type,
            is_public=is_public,
            tasks=tasks,
            training_image=training_image,
            training_model_config=training_model_config,
            training_loader_config=training_loader_config,
            training_hyperparameter_config=training_hyperparameter_config,
            performance_ratings=performance_ratings,
            config_file_path=config_file_path,
        )
    )
    _print_architecture_info(architecture)


def update_architecture(
    identifier: UUID | str,
    *,
    name: str | None = None,
    description: str | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    tasks: list[Task] | None = None,
    training_image: str | None = None,
    training_model_config: dict[str, Any] | None = None,
    training_loader_config: dict[str, Any] | None = None,
    training_hyperparameter_config: dict[str, Any] | None = None,
    performance_ratings: PerformanceRatings | dict[str, Any] | None = None,
    config_file_path: str | None = None,
) -> ArchitectureResponse:
    """Updates an architecture resource.

    Parameters
    ----------
    identifier : UUID | str
        The architecture ID or slug.
    name : str | None
        The name of the architecture.
    description : str | None
        Full description of the architecture.
    license_type : License | None
        The type of the license.
    is_public : bool | None
        Visibility change; ``None`` leaves it unchanged. The server rejects
        promotion to public through this endpoint.
    tasks : list[Task] | None
        List of tasks this architecture supports.
    training_image : str | None
        The internal training image used in this architecture (DO NOT expose to user).
    training_model_config : dict[str, Any] | None
        The `model` configuration block parsed by the training image.
    training_loader_config : dict[str, Any] | None
        The optional `loader` configuration block parsed by the training image.
    training_hyperparameter_config : dict[str, Any] | None
        The `trainer` configuration block parsed by the training image.
    performance_ratings : PerformanceRatings | dict[str, Any] | None
        The speed and accuracy ratings for each HW with this architecture.
    config_file_path : str | None
        Path to a JSON or YAML config file that is used in the luxonis-train for defining the model.
    """

    if isinstance(identifier, UUID):
        identifier = str(identifier)
    architecture = get_architecture(identifier)
    updates = {
        key: value
        for key, value in {
            "name": name,
            "description": description,
            "license_type": license_type,
            "is_public": is_public,
            "tasks": tasks,
            "training_image": training_image,
            "training_model_config": training_model_config,
            "training_loader_config": training_loader_config,
            "training_hyperparameter_config": training_hyperparameter_config,
            "performance_ratings": performance_ratings,
        }.items()
        if value is not None
    }
    if config_file_path is not None:
        updates["config_file_path"] = config_file_path
    if not updates:
        return architecture

    existing = architecture.model_dump(mode="json")
    data = {
        key: existing[key]
        for key in ArchitectureConfig.model_fields
        if key in existing and key != "config_file_path"
    }
    if config_file_path is not None:
        for key in (
            "training_model_config",
            "training_loader_config",
            "training_hyperparameter_config",
        ):
            data.pop(key, None)
    data.update(updates)
    try:
        architecture_config = ArchitectureConfig(**data)
    except PydanticValidationError as exc:
        raise _config_input_error(exc) from exc
    patch_fields = set(updates) - {"config_file_path"}
    if config_file_path is not None:
        patch_fields.update(
            {
                "training_model_config",
                "training_loader_config",
                "training_hyperparameter_config",
            }
        )
    patch_data = architecture_config.model_dump(
        mode="json", include=patch_fields
    )

    try:
        res = Request.patch(
            service="models",
            endpoint=f"architectures/{architecture.id}",
            json=patch_data,
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc,
            identifier=identifier,
            endpoint="architectures",
            conflict_message=(
                f"Architecture '{name or identifier}' already exists"
            ),
        )

    logger.info(f"Architecture '{identifier}' updated successfully.")

    return ArchitectureResponse(**res)


@app.command(name="update")
def update_architecture_cli(
    identifier: UUID | str,
    *,
    name: str | None = None,
    description: str | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    tasks: list[Task] | None = None,
    training_image: str | None = None,
    training_model_config: TrainingModelConfigOption = None,
    training_loader_config: TrainingLoaderConfigOption = None,
    training_hyperparameter_config: TrainingHyperparameterConfigOption = None,
    performance_ratings: PerformanceRatings | dict[str, Any] | None = None,
    config_file_path: ConfigFilePathOption = None,
) -> None:
    """Updates an architecture resource."""
    architecture = run_cli(
        lambda: update_architecture(
            identifier,
            name=name,
            description=description,
            license_type=license_type,
            is_public=is_public,
            tasks=tasks,
            training_image=training_image,
            training_model_config=training_model_config,
            training_loader_config=training_loader_config,
            training_hyperparameter_config=training_hyperparameter_config,
            performance_ratings=performance_ratings,
            config_file_path=config_file_path,
        )
    )
    _print_architecture_info(architecture)


def delete_architecture(identifier: UUID | str) -> None:
    """Delete an architecture.

    Parameters
    ----------
    identifier : UUID | str
        The architecture ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    architecture_id = resolve_resource_id(identifier, "architectures")
    try:
        Request.delete(
            service="models", endpoint=f"architectures/{architecture_id}"
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc,
            identifier=identifier,
            endpoint="architectures",
        )
    logger.info(f"Architecture '{identifier}' deleted")


@app.command(name="delete")
def delete_architecture_cli(identifier: UUID | str) -> None:
    """Delete an architecture."""
    run_cli(lambda: delete_architecture(identifier))


def _print_architecture_list(
    architectures: list[ArchitectureResponse],
    field: list[str] | None = None,
) -> None:
    print_hub_ls(
        [
            _architecture_to_cli_data(architecture)
            for architecture in architectures
        ],
        keys=field or ARCHITECTURE_LIST_KEYS,
    )


def _print_architecture_info(architecture: ArchitectureResponse) -> None:
    print_hub_resource_info(
        _architecture_to_cli_data(architecture),
        title="Architecture Info",
        json=False,
        keys=ARCHITECTURE_INFO_KEYS,
    )


def _architecture_to_cli_data(
    architecture: ArchitectureResponse,
) -> dict[str, object]:
    return architecture.model_dump(mode="json")


def _config_input_error(exc: PydanticValidationError) -> InputError:
    errors = exc.errors()
    if errors:
        message = errors[0].get("msg")
        if isinstance(message, str):
            return InputError(message.removeprefix("Value error, "))
    return InputError(str(exc))
