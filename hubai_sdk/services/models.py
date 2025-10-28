from typing import Annotated
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger

from hubai_sdk.typing import License, Order, Task
from hubai_sdk.utils.general import is_cli_call
from hubai_sdk.utils.hub import (
    get_resource_id,
    hub_ls,
    print_hub_resource_info,
    request_info,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import ModelResponse

app = App(
    name="model", help="Models Interactions", group="Resource Management"
)


@app.command(name="ls")
def list_models(
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    slug: str | None = None,
    project_id: str | None = None,
    luxonis_only: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelResponse] | None:
    silent = not is_cli_call()
    data = hub_ls(
        "models",
        tasks=list(tasks) if tasks else [],
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        luxonis_only=luxonis_only,
        limit=limit,
        sort=sort,
        order=order,
        _silent=silent,
        keys=field or ["name", "id", "slug"],
    )

    if not silent:
        return None

    return [ModelResponse(**model) for model in data]


@app.command(name="info")
def get_model(identifier: UUID | str) -> ModelResponse | None:
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    silent = not is_cli_call()
    data = request_info(identifier, "models")

    if not silent:
        return print_hub_resource_info(
            data,
            title="Model Info",
            json=False,
            keys=[
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
                "team_id",
            ],
        )
    return ModelResponse(**data)


@app.command(name="create")
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
    silent: bool | None = None,
) -> ModelResponse | None:
    """Creates a new model resource.

    Parameters
    ----------
    name : str
        The name of the model.
    license_type : License
        The type of the license.
    is_public : bool | None
        Whether the model is public (True), private (False), or team (None).
    description : str | None
        Full description of the model.
    description_short : str
        Short description of the model.
    architecture_id : UUID | str | None
        The architecture ID.
    tasks : list[Task] | None
        List of tasks this model supports.
    links : list[str] | None
        List of links to related resources.
    is_yolo : bool
        Whether the model is a YOLO model.
    silent : bool | None
        Whether to print the model information after creation.
    """

    if silent is None:
        silent = not is_cli_call()
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
    except requests.HTTPError as e:
        if (
            e.response is not None
            and e.response.json().get("detail") == "Unique constraint error."
        ):
            raise ValueError(f"Model '{name}' already exists") from e
        raise
    logger.info(f"Model '{res['name']}' created with ID '{res['id']}'")

    if not silent:
        return get_model(res["id"])
    return ModelResponse(**res)


@app.command(name="delete")
def delete_model(identifier: UUID | str) -> None:
    """Deletes a model.

    Parameters
    ----------
    identifier : UUID | str
        The model ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    model_id = get_resource_id(identifier, "models")
    Request.delete(service="models", endpoint=f"models/{model_id}")
    logger.info(f"Model '{identifier}' deleted")
