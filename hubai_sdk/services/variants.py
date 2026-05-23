from typing import Annotated
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger

from hubai_sdk.typing import Order
from hubai_sdk.utils.hub import (
    get_resource_info,
    print_hub_ls,
    print_hub_resource_info,
    raise_for_hub_error,
    resolve_resource_id,
    run_cli,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.sdk_models import ModelVersionResponse
from hubai_sdk.utils.telemetry import get_telemetry

app = App(
    name="variant",
    help="Model variants Interactions",
    group="Resource Management",
)

VARIANT_LIST_KEYS = ["name", "version", "id", "platforms"]
VARIANT_LIST_KEYS_WITH_MODEL = [
    "model_name",
    "name",
    "version",
    "id",
    "platforms",
]
VARIANT_INFO_KEYS = [
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
]


def list_variants(
    model_id: UUID | str | None = None,
    name: str | None = None,
    variant_slug: str | None = None,
    variant_version: str | None = None,
    is_public: bool | None = None,
    include_model_name: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
) -> list[ModelVersionResponse]:
    """List the model versions in the HubAI.

    Parameters
    ----------
    model_id : UUID | str | None
        Filter the listed model versions by model ID.
    name : str | None
        Filter the listed model versions by name.
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
    """

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "variants.list",
            properties={"model_id": model_id},
            include_system_metadata=False,
        )

    try:
        data = Request.get(
            service="models",
            endpoint="modelVersions",
            params={
                "model_id": str(model_id) if model_id else None,
                "name": name,
                "variant_slug": variant_slug,
                "version": variant_version,
                "is_public": is_public,
                "limit": limit,
                "sort": sort,
                "order": order,
            },
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(exc)

    if include_model_name:
        for variant in data:
            variant["model_name"] = get_resource_info(
                variant["model_id"], "models"
            )["name"]

    return [ModelVersionResponse(**variant) for variant in data]


@app.command(name="ls")
def list_variants_cli(
    model_id: UUID | str | None = None,
    name: str | None = None,
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
) -> None:
    """List the model versions in the HubAI."""
    variants = run_cli(
        lambda: list_variants(
            model_id=model_id,
            name=name,
            variant_slug=variant_slug,
            variant_version=variant_version,
            is_public=is_public,
            include_model_name=include_model_name,
            limit=limit,
            sort=sort,
            order=order,
        )
    )
    _print_variant_list(variants, include_model_name, field)


def get_variant(identifier: UUID | str) -> ModelVersionResponse:
    """Returns information about a model version.

    Parameters
    ----------
    identifier : UUID | str
        The model version ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    data = get_resource_info(identifier, "modelVersions")

    data["model_name"] = get_resource_info(data["model_id"], "models")["name"]

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "variants.get",
            properties={"variant_id": identifier},
            include_system_metadata=False,
        )

    return ModelVersionResponse(**data)


@app.command(name="info")
def get_variant_info(identifier: UUID | str) -> None:
    """Returns information about a model version."""
    _print_variant_info(run_cli(lambda: get_variant(identifier)))


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
) -> ModelVersionResponse:
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
    """

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
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc,
            conflict_message=(
                f"Model variant '{name}' already exists for model '{model_id}'"
            ),
        )

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "variants.create", properties=data, include_system_metadata=False
        )

    logger.info(f"Model variant '{res['name']}' created with ID '{res['id']}'")

    return get_variant(res["id"])


@app.command(name="create")
def create_variant_cli(
    name: str,
    *,
    model_id: UUID | str,
    variant_version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Creates a new variant of a model."""
    variant = run_cli(
        lambda: create_variant(
            name,
            model_id=model_id,
            variant_version=variant_version,
            description=description,
            repository_url=repository_url,
            commit_hash=commit_hash,
            domain=domain,
            tags=tags,
        )
    )
    _print_variant_info(variant)


def delete_variant(identifier: UUID | str) -> None:
    """Deletes a model variant.

    Parameters
    ----------
    identifier : UUID | str
        The model variant ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    variant_id = resolve_resource_id(identifier, "modelVersions")
    try:
        Request.delete(
            service="models", endpoint=f"modelVersions/{variant_id}"
        )
    except requests.HTTPError as exc:
        raise_for_hub_error(
            exc, identifier=identifier, endpoint="modelVersions"
        )
    logger.info(f"Model variant '{variant_id}' deleted")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture(
            "variants.delete",
            properties={"variant_id": identifier},
            include_system_metadata=False,
        )


@app.command(name="delete")
def delete_variant_cli(identifier: UUID | str) -> None:
    """Deletes a model variant."""
    run_cli(lambda: delete_variant(identifier))


def _print_variant_list(
    variants: list[ModelVersionResponse],
    include_model_name: bool,
    field: list[str] | None = None,
) -> None:
    print_hub_ls(
        [_variant_to_cli_data(variant) for variant in variants],
        keys=field
        or (
            VARIANT_LIST_KEYS_WITH_MODEL
            if include_model_name
            else VARIANT_LIST_KEYS
        ),
    )


def _print_variant_info(variant: ModelVersionResponse) -> None:
    print_hub_resource_info(
        _variant_to_cli_data(variant),
        title="Model Variant Info",
        json=False,
        keys=VARIANT_INFO_KEYS,
    )


def _variant_to_cli_data(variant: ModelVersionResponse) -> dict[str, object]:
    return variant.model_dump(mode="json")
