from typing import Annotated
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger

from hubai_sdk.typing import License, Order, Task
from hubai_sdk.utils.hub import (
    get_resource_info,
    print_hub_ls,
    print_hub_resource_info,
    raise_for_hub_error,
    resolve_resource_id,
    run_cli,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.sdk_models import ModelResponse
from hubai_sdk.utils.telemetry import get_telemetry

app = App(
    name="model", help="Models Interactions", group="Resource Management"
)

MODEL_LIST_KEYS = ["name", "id", "slug"]
MODEL_INFO_KEYS = [
    "name",
    "slug",
    "id",
    "created",
    "updated",
    "tasks",
    "platforms",
    "is_public",
    "is_commercial",
    "license_type",
    "versions",
    "likes",
    "downloads",
]


def list_models(
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    project_id: str | None = None,
    luxonis_only: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
) -> list[ModelResponse]:
    """List the models in the HubAI.

    Args:
        tasks: list[Task] | None. Filter the listed models by tasks.
        license_type: License | None. Filter the listed models by license type.
        is_public: bool | None. Filter the listed models by public status.
        project_id: str | None. Filter the listed models by project ID.
        luxonis_only: bool. Filter the listed models by Luxonis only.
        limit: int. Maximum number of models to return.
        sort: str. Field to sort the models by. It should be the field name from the ModelResponse. For example, "name", "id", "updated", etc.
        order: Order. Order to sort the models by. It should be "asc" or "desc".

    Returns:
        A list of matching model resources.
    """
    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("models.list", include_system_metadata=False)

    try:
        data = Request.get(
            service="models",
            endpoint="models",
            params={
                "tasks": tasks,
                "license_type": license_type,
                "is_public": is_public,
                "project_id": project_id,
                "luxonis_only": luxonis_only,
                "limit": limit,
                "sort": sort,
                "order": order,
            },
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(exc)

    return [ModelResponse(**model) for model in data]


@app.command(name="ls")
def list_models_cli(
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    project_id: str | None = None,
    luxonis_only: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    """List the models in the HubAI."""
    models = run_cli(
        lambda: list_models(
            tasks=tasks,
            license_type=license_type,
            is_public=is_public,
            project_id=project_id,
            luxonis_only=luxonis_only,
            limit=limit,
            sort=sort,
            order=order,
        )
    )
    _print_model_list(models, field)


def get_model(identifier: UUID | str) -> ModelResponse:
    """Get the model information from the HubAI.

    Args:
        identifier: UUID | str. The model ID or slug.

    Returns:
        The resolved model resource.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    data = get_resource_info(identifier, "models")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "models.get",
            properties={"model_id": identifier},
            include_system_metadata=False,
        )

    return ModelResponse(**data)


@app.command(name="info")
def get_model_info_cli(identifier: UUID | str) -> None:
    """Get the model information from the HubAI."""
    _print_model_info(run_cli(lambda: get_model(identifier)))


def create_model(
    name: str,
    *,
    license_type: License = "undefined",
    is_public: bool | None = False,
    description: str | None = None,
    description_short: str = "<empty>",
    architecture_id: UUID | str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool = False,
) -> ModelResponse:
    """Creates a new model resource.

    Args:
        name: str. The name of the model.
        license_type: License. The type of the license.
        is_public: bool | None. Whether the model is public (True), private (False), or team (None).
        description: str | None. Full description of the model.
        description_short: str. Short description of the model.
        architecture_id: UUID | str | None. The architecture ID.
        tasks: list[Task] | None. List of tasks this model supports.
        links: list[str] | None. List of links to related resources.
        is_yolo: bool. Whether the model is a YOLO model.

    Returns:
        The created model resource.
    """
    data = {
        "name": name,
        "license_type": license_type,
        "is_public": is_public,
        "description_short": description_short,
        "description": description,
        "architecture_id": str(architecture_id) if architecture_id else None,
        "tasks": tasks or [],
        "links": links or [],
        "is_yolo": is_yolo,
    }
    try:
        res = Request.post(service="models", endpoint="models", json=data)
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc, conflict_message=f"Model '{name}' already exists"
        )
    logger.info(f"Model '{res['name']}' created with ID '{res['id']}'")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "models.create", properties=data, include_system_metadata=False
        )

    return get_model(res["id"])


@app.command(name="create")
def create_model_cli(
    name: str,
    *,
    license_type: License = "undefined",
    is_public: bool | None = False,
    description: str | None = None,
    description_short: str = "<empty>",
    architecture_id: UUID | str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool = False,
) -> None:
    """Creates a new model resource."""
    model = run_cli(
        lambda: create_model(
            name,
            license_type=license_type,
            is_public=is_public,
            description=description,
            description_short=description_short,
            architecture_id=architecture_id,
            tasks=tasks,
            links=links,
            is_yolo=is_yolo,
        )
    )
    _print_model_info(model)


def update_model(
    identifier: UUID | str,
    *,
    license_type: License | None = None,
    is_public: bool | None = None,
    description: str | None = None,
    description_short: str | None = None,
    architecture_id: UUID | str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool | None = None,
) -> ModelResponse:
    """Updates a model.

    Args:
        identifier: UUID | str. The model ID or slug.
        license_type: License | None. The type of the license.
        is_public: bool | None. Whether the model is public (True), private (False), or team (None).
        description: str | None. Full description of the model.
        description_short: str | None. Short description of the model.
        architecture_id: UUID | str | None. The architecture ID.
        tasks: list[Task] | None. List of tasks this model supports.
        links: list[str] | None. List of links to related resources.
        is_yolo: bool | None. Whether the model is a YOLO model.

    Returns:
        The updated model resource.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    model_id = resolve_resource_id(identifier, "models")

    data = {}
    if license_type is not None:
        data["license_type"] = license_type
    if is_public is not None:
        data["is_public"] = is_public
    if description is not None:
        data["description"] = description
    if description_short is not None:
        data["description_short"] = description_short
    if architecture_id is not None:
        data["architecture_id"] = str(architecture_id)
    if tasks is not None:
        data["tasks"] = tasks
    if links is not None:
        data["links"] = links
    if is_yolo is not None:
        data["is_yolo"] = is_yolo
    try:
        res = Request.patch(
            service="models", endpoint=f"models/{model_id}", json=data
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc,
            identifier=identifier,
            endpoint="models",
            conflict_message=f"Model '{identifier}' already exists",
        )
    logger.info(f"Model '{res['name']}' updated with ID '{res['id']}'")

    telemetry = get_telemetry()
    if telemetry:
        data["model_id"] = identifier
        telemetry.capture(
            "models.update", properties=data, include_system_metadata=False
        )

    return get_model(res["id"])


@app.command(name="update")
def update_model_cli(
    identifier: UUID | str,
    *,
    license_type: License | None = None,
    is_public: bool | None = None,
    description: str | None = None,
    description_short: str | None = None,
    architecture_id: UUID | str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool | None = None,
) -> None:
    """Updates a model."""
    model = run_cli(
        lambda: update_model(
            identifier,
            license_type=license_type,
            is_public=is_public,
            description=description,
            description_short=description_short,
            architecture_id=architecture_id,
            tasks=tasks,
            links=links,
            is_yolo=is_yolo,
        )
    )
    _print_model_info(model)


def delete_model(identifier: UUID | str) -> None:
    """Deletes a model.

    Args:
        identifier: UUID | str. The model ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    model_id = resolve_resource_id(identifier, "models")
    try:
        Request.delete(service="models", endpoint=f"models/{model_id}")
    except requests.HTTPError as exc:
        raise_for_hub_error(exc, identifier=identifier, endpoint="models")
    logger.info(f"Model '{identifier}' deleted")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "models.delete",
            properties={"model_id": identifier},
            include_system_metadata=False,
        )


@app.command(name="delete")
def delete_model_cli(identifier: UUID | str) -> None:
    """Deletes a model."""
    run_cli(lambda: delete_model(identifier))


def _print_model_list(
    models: list[ModelResponse], field: list[str] | None = None
) -> None:
    print_hub_ls(
        [_model_to_cli_data(model) for model in models],
        keys=field or MODEL_LIST_KEYS,
    )


def _print_model_info(model: ModelResponse) -> None:
    print_hub_resource_info(
        _model_to_cli_data(model),
        title="Model Info",
        json=False,
        keys=MODEL_INFO_KEYS,
    )


def _model_to_cli_data(model: ModelResponse) -> dict[str, object]:
    return model.model_dump(mode="json")
