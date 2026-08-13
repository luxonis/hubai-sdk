"""Create and manage top-level model resources.

A model contains shared metadata such as its display name, supported tasks,
license, and visibility. Actual files belong to model instances beneath a
versioned variant; see `hubai_sdk.services.variants` and
`hubai_sdk.services.instances`.

Identifiers:
    Functions that accept an ``identifier`` support a UUID, a raw model slug,
    or a ``<team>/<model>`` resource path. This makes slugs suitable for
    configuration files while UUIDs remain useful for programmatic flows.

Visibility:
    During creation, ``is_public=True`` makes a model public, ``False`` makes
    it private, and ``None`` selects team visibility. For list operations,
    ``None`` means no visibility filter. For updates, ``None`` leaves the
    existing visibility unchanged.

Example:
    .. python::

        models = client.models.list_models(
            tasks=["OBJECT_DETECTION"],
            is_public=True,
            limit=10,
        )
        model = client.models.get_model(models[0].id)
"""

from typing import Annotated
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger
from luxonis_ml.telemetry import suppress_telemetry

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
from hubai_sdk.utils.telemetry import (
    MODEL_CREATED_EVENT,
    MODEL_DELETED_EVENT,
    MODEL_RETRIEVED_EVENT,
    MODEL_UPDATED_EVENT,
    MODELS_LISTED_EVENT,
    OperationName,
    OperationTelemetrySpec,
    TargetResource,
    TelemetryGroup,
    build_model_created_properties,
    build_model_identifier_properties,
    build_model_updated_properties,
    build_models_listed_properties,
    telemetry_operation,
)

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


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.MODELS_LIST,
        operation_group=TelemetryGroup.MODELS,
        success_event=MODELS_LISTED_EVENT,
        target_resource=TargetResource.MODEL,
        success_builder=build_models_listed_properties,
    )
)
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
    """List models visible to the authenticated HubAI team.

    Args:
        tasks: Filter models by supported tasks.
        license_type: Keep models with this license.
        is_public: Filter by public status. Leave as ``None`` to include every
            visibility available to the API key.
        project_id: Keep models belonging to this project.
        luxonis_only: Whether to return only Luxonis-maintained models.
        limit: Maximum number of models to return.
        sort: `ModelResponse` field used for sorting, such as ``"name"``,
            ``"id"``, or ``"updated"``.
        order: Sort in ascending or descending order.

    Returns:
        A list of matching model resources.
    """
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


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.MODEL_GET,
        operation_group=TelemetryGroup.MODELS,
        success_event=MODEL_RETRIEVED_EVENT,
        target_resource=TargetResource.MODEL,
        identifier_param="identifier",
        success_builder=build_model_identifier_properties,
    )
)
def get_model(identifier: UUID | str) -> ModelResponse:
    """Get one model by identifier.

    Args:
        identifier: Model UUID, raw slug, or ``<team>/<model>`` path.

    Returns:
        The resolved model resource.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    data = get_resource_info(identifier, "models")

    return ModelResponse(**data)


@app.command(name="info")
def get_model_info_cli(identifier: UUID | str) -> None:
    """Get the model information from the HubAI."""
    _print_model_info(run_cli(lambda: get_model(identifier)))


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.MODEL_CREATE,
        operation_group=TelemetryGroup.MODELS,
        success_event=MODEL_CREATED_EVENT,
        target_resource=TargetResource.MODEL,
        success_builder=build_model_created_properties,
    )
)
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
    """Create a model resource.

    Args:
        name: Human-readable model name. HubAI derives the slug from it.
        license_type: License attached to the model metadata.
        is_public: ``True`` for public, ``False`` for private, or ``None`` for
            team visibility.
        description: Full model description.
        description_short: Short summary shown in model listings.
        architecture_id: Related architecture UUID, if known.
        tasks: Tasks supported by the model.
        links: URLs for source code, papers, or other related resources.
        is_yolo: Whether the model uses a supported YOLO output contract.

    Returns:
        The created model resource.

    Raises:
        ResourceConflictError: If a model with the derived slug already
            exists.
        HubApiError: If HubAI rejects the request for another reason.
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

    with suppress_telemetry():
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


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.MODEL_UPDATE,
        operation_group=TelemetryGroup.MODELS,
        success_event=MODEL_UPDATED_EVENT,
        target_resource=TargetResource.MODEL,
        identifier_param="identifier",
        success_builder=build_model_updated_properties,
    )
)
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
    """Update fields on an existing model.

    Only arguments whose value is not ``None`` are sent to HubAI. This means
    optional text fields cannot be cleared with this helper.

    Args:
        identifier: Model UUID, raw slug, or ``<team>/<model>`` path.
        license_type: Replacement license.
        is_public: Replacement public/private state.
        description: Replacement full description.
        description_short: Replacement short description.
        architecture_id: Replacement architecture UUID.
        tasks: Replacement task list.
        links: Replacement related-resource links.
        is_yolo: Replacement YOLO flag.

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

    with suppress_telemetry():
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


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.MODEL_DELETE,
        operation_group=TelemetryGroup.MODELS,
        success_event=MODEL_DELETED_EVENT,
        target_resource=TargetResource.MODEL,
        identifier_param="identifier",
        success_builder=build_model_identifier_properties,
    )
)
def delete_model(identifier: UUID | str) -> None:
    """Delete a model from HubAI.

    Args:
        identifier: Model UUID, raw slug, or ``<team>/<model>`` path.

    Raises:
        ResourceNotFoundError: If ``identifier`` cannot be resolved.
        HubApiError: If HubAI refuses the deletion.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    model_id = resolve_resource_id(identifier, "models")
    try:
        Request.delete(service="models", endpoint=f"models/{model_id}")
    except requests.HTTPError as exc:
        raise_for_hub_error(exc, identifier=identifier, endpoint="models")
    logger.info(f"Model '{identifier}' deleted")


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
