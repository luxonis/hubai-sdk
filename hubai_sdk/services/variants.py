from typing import Annotated, overload
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger

from hubai_sdk.typing import Order
from hubai_sdk.utils.general import is_cli_call
from hubai_sdk.utils.hub import (
    get_resource_id,
    print_hub_ls,
    print_hub_resource_info,
    request_info,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.sdk_models import ModelVersionResponse
from hubai_sdk.utils.telemetry import get_telemetry

app = App(
    name="variant",
    help="Model variants Interactions",
    group="Resource Management",
)

@overload
def list_variants(
    model_id: UUID | str | None = None,
    variant_slug: str | None = None,
    variant_version: str | None = None,
    is_public: bool | None = None,
    include_model_name: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelVersionResponse]:
    ...

@overload
def list_variants(
    model_id: UUID | str | None = None,
    variant_slug: str | None = None,
    variant_version: str | None = None,
    is_public: bool | None = None,
    include_model_name: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelVersionResponse]:
    ...

@app.command(name="ls")
def list_variants(
    model_id: UUID | str | None = None,
    variant_slug: str | None = None,
    variant_version: str | None = None,
    is_public: bool | None = None,
    include_model_name: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelVersionResponse] | None:
    """List the model versions in the HubAI.

    Parameters
    ----------
    model_id : UUID | str | None
        Filter the listed model versions by model ID.
    variant_slug : str | None
        Filter the listed model versions by variant slug.
    variant_version : str | None
        Filter the listed model versions by version.
    is_public : bool | None
        Filter the listed model versions by visibility.
    include_model_name : bool
        Whether to include the model name in the response. By default, it is False and the ModelVersionResponse will have "model_name" field as None. If True, the ModelVersionResponse will have "model_name" field as the name of the model.
    limit : int
        Limit the number of model versions to show.
    sort : str
        Sort the model versions by this field. It should be the field name from the ModelVersionResponse. For example, "name", "id", "updated", etc.
    order : Literal["asc", "desc"]
        Order to sort the model versions by. It should be "asc" or "desc".
    field : list[str] | None
        Fields to include in the response in case of CLI usage.
        By default, ["name", "version", "slug", "platforms"] are shown. If include_model_name is True, ["model_name"] is added.
    """

    silent = not is_cli_call()

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("variants.list", properties={"model_id": model_id}, include_system_metadata=False)

    data = Request.get(service="models", endpoint="modelVersions", params={
        "model_id": str(model_id) if model_id else None,
        "variant_slug": variant_slug,
        "version": variant_version,
        "is_public": is_public,
        "limit": limit,
        "sort": sort,
        "order": order,
    })

    if include_model_name:
        for variant in data:
            variant["model_name"] = request_info(variant["model_id"], "models")["name"]

    if not silent:
        return print_hub_ls(
            data,
            keys=field or (["name", "version", "slug", "platforms"] if not include_model_name else ["model_name", "name", "version", "slug", "platforms"]),
            silent=silent
        )

    return [ModelVersionResponse(**variant) for variant in data]


@overload
def get_variant(identifier: UUID | str, silent: bool | None = None) -> ModelVersionResponse:
    ...

@overload
def get_variant(identifier: UUID | str, silent: bool | None = None) -> None:
    ...

@app.command(name="info")
def get_variant(identifier: UUID | str, silent: bool | None = None) -> ModelVersionResponse | None:
    """Returns information about a model version.

    Parameters
    ----------
    identifier : UUID | str
        The model version ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    if silent is None:
        silent = not is_cli_call()
    data = request_info(identifier, "modelVersions")

    data["model_name"] = request_info(data["model_id"], "models")["name"]

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("variants.get", properties={"variant_id": identifier}, include_system_metadata=False)

    if not silent:
        return print_hub_resource_info(
            data,
            title="Model Variant Info",
            json=False,
            keys=[
                "model_name",
                "name",
                "slug",
                "version",
                "id",
                "model_id",
                "created",
                "updated",
                "platforms",
                "exportable_to",
                "is_public",
            ],
        )
    return ModelVersionResponse(**data)

@overload
def create_variant(
    name: str,
    *,
    model_id: UUID | str,
    variant_version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
    silent: bool = True
) -> ModelVersionResponse:
    ...

@overload
def create_variant(
    name: str,
    *,
    model_id: UUID | str,
    variant_version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
    silent: bool = False,
) -> None:
    ...

@overload
def create_variant(
    name: str,
    *,
    model_id: UUID | str,
    variant_version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
    silent: bool | None = None,
) -> ModelVersionResponse:
    ...

@app.command(name="create")
def create_variant(
    name: str,
    *,
    model_id: UUID | str,
    variant_version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
    silent: bool | None = None,
) -> ModelVersionResponse | None:
    """Creates a new variant of a model.

    Parameters
    ----------
    name : str
        The name of the model variant.
    model_id : UUID | str
        The ID of the model to create a variant for.
    variant_version : str
        The version of the model variant.
    description : str | None
        Full description of the model variant.
    repository_url : str | None
        URL of the related repository.
    commit_hash : str | None
        Commit hash.
    domain : str | None
        Domain of the model variant.
    tags : list[str] | None
        List of tags for the model variant.
    silent : bool
        Whether to print the model variant information after creation.
    """

    if silent is None:
        silent = not is_cli_call()

    data = {
        "model_id": str(model_id) if model_id else None,
        "name": name,
        "version": variant_version,
        "description": description,
        "repository_url": repository_url,
        "commit_hash": commit_hash,
        "domain": domain,
        "tags": tags or [],
    }

    try:
        res = Request.post(
            service="models", endpoint="modelVersions", json=data
        )
    except requests.HTTPError as e:
        if str(e).startswith("{'detail': 'Unique constraint error."):
            raise ValueError(
                f"Model variant '{name}' already exists for model '{model_id}'"
            ) from e
        raise

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("variants.create", properties=data, include_system_metadata=False)

    logger.info(f"Model variant '{res['name']}' created with ID '{res['id']}'")

    return get_variant(res["id"], silent)


@app.command(name="delete")
def delete_variant(identifier: UUID | str) -> None:
    """Deletes a model variant.

    Parameters
    ----------
    identifier : UUID | str
        The model variant ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    variant_id = get_resource_id(identifier, "modelVersions")
    Request.delete(service="models", endpoint=f"modelVersions/{variant_id}")
    logger.info(f"Model variant '{variant_id}' deleted")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("variants.delete", properties={"variant_id": identifier}, include_system_metadata=False)
