import signal
from pathlib import Path
from types import FrameType
from typing import Annotated, Literal, overload
from urllib.parse import unquote, urlparse
from uuid import UUID

import requests
from cyclopts import App, Parameter
from loguru import logger
from rich.progress import Progress

from hubai_sdk.typing import (
    ModelClass,
    Order,
    Quantization,
    Status,
    TargetPrecision,
    YoloVersion,
)
from hubai_sdk.utils.general import is_cli_call
from hubai_sdk.utils.hub import (
    get_resource_id,
    print_hub_ls,
    print_hub_resource_info,
    request_info,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import (
    ArchiveConfigurationResponse,
    ModelInstanceFileResponse,
)
from hubai_sdk.utils.sdk_models import ModelInstanceResponse
from hubai_sdk.utils.types import ModelType
from hubai_sdk.utils.telemetry import get_telemetry

app = App(
    name="instance",
    help="Model instances Interactions",
    group="Resource Management",
)

@overload
def list_instances(
    *,
    platforms: list[ModelType] | None = None,
    model_id: UUID | str | None = None,
    variant_id: UUID | str | None = None,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_class: ModelClass | None = None,
    name: str | None = None,
    hash: str | None = None,
    status: Status | None = None,
    is_public: bool | None = None,
    compression_level: Literal[0, 1, 2, 3, 4, 5] | None = None,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] | None = None,
    slug: str | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelInstanceResponse]:
    ...

@overload
def list_instances(
    *,
    platforms: list[ModelType] | None = None,
    model_id: UUID | str | None = None,
    variant_id: UUID | str | None = None,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_class: ModelClass | None = None,
    name: str | None = None,
    hash: str | None = None,
    status: Status | None = None,
    is_public: bool | None = None,
    compression_level: Literal[0, 1, 2, 3, 4, 5] | None = None,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] | None = None,
    slug: str | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    ...

@app.command(name="ls")
def list_instances(
    *,
    platforms: list[ModelType] | None = None,
    model_id: UUID | str | None = None,
    variant_id: UUID | str | None = None,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_class: ModelClass | None = None,
    name: str | None = None,
    hash: str | None = None,
    status: Status | None = None,
    is_public: bool | None = None,
    compression_level: Literal[0, 1, 2, 3, 4, 5] | None = None,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] | None = None,
    slug: str | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> list[ModelInstanceResponse] | None:
    """List the model instances in the HubAI.

    Parameters
    ----------
    platforms : list[ModelType] | None
        Filter the listed model instances by platforms.
    model_id : UUID | str | None
        Filter the listed model instances by model ID.
    variant_id : UUID | str | None
        Filter the listed model instances by variant ID.
    model_type : ModelType | None
        Filter the listed model instances by model type.
    parent_id : UUID | str | None
        Filter the listed model instances by parent ID.
    model_class : ModelClass | None
        Filter the listed model instances by model class.
    name : str | None
        Filter the listed model instances by name.
    hash : str | None
        Filter the listed model instances by hash.
    status : Status | None
        Filter the listed model instances by status.
    is_public : bool | None
        Filter the listed model instances by visibility.
    compression_level : Literal[0, 1, 2, 3, 4, 5] | None
        Filter the listed model instances by compression level.
        Only relevant for Hailo models.
    optimization_level : Literal[-100, 0, 1, 2, 3, 4] | None
        Filter the listed model instances by optimization level.
        Only relevant for Hailo models.
    slug : str | None
        Filter the listed model instances by slug.
    limit : int
        Limit the number of model instances to show.
    sort : str
        Sort the model instances by this field. It should be the field name from the ModelInstanceResponse. For example, "name", "id", "updated", etc.
    order : Literal["asc", "desc"]
        Order to sort the model instances by. It should be "asc" or "desc".
    field : list[str] | None
        Fields to include in the response in case of CLI usage.
        By default, ["slug", "id", "model_type", "is_nn_archive", "model_precision_type"] are shown.
    """

    silent = not is_cli_call()

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.list", include_system_metadata=True)

    data = Request.get(service="models", endpoint="modelInstances", params={
        "platforms": [platform.name for platform in platforms] if platforms else [],
        "model_id": str(model_id) if model_id else None,
        "model_version_id": str(variant_id) if variant_id else None,
        "model_type": model_type,
        "parent_id": str(parent_id) if parent_id else None,
        "model_class": model_class,
        "name": name,
        "hash": hash,
        "status": status,
        "compression_level": compression_level,
        "optimization_level": optimization_level,
        "is_public": is_public,
        "slug": slug,
        "limit": limit,
        "sort": sort,
        "order": order,
    })

    for instance in data:
        instance["model_name"] = request_info(instance["model_id"], "models")["name"]
        instance["model_variant_name"] = request_info(instance["model_version_id"], "modelVersions")["name"]

    if not silent:
        return print_hub_ls(
            data,
            keys=field or ["model_name", "model_variant_name", "slug", "id", "model_type", "is_nn_archive", "model_precision_type"],
            silent=silent
        )

    return [ModelInstanceResponse(**instance) for instance in data]


@overload
def get_instance(identifier: UUID | str) -> ModelInstanceResponse:
    ...

@overload
def get_instance(identifier: UUID | str) -> None:
    ...

@app.command(name="info")
def get_instance(identifier: UUID | str) -> ModelInstanceResponse | None:
    """Returns information about a model instance.

    Parameters
    ----------
    identifier : UUID | str
        The model instance ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    silent = not is_cli_call()
    data = request_info(identifier, "modelInstances")

    data["model_name"] = request_info(data["model_id"], "models")["name"]
    data["model_variant_name"] = request_info(data["model_version_id"], "modelVersions")["name"]

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.get", properties={"instance_id": identifier}, include_system_metadata=True)

    if not silent:
        return print_hub_resource_info(
            data,
            title="Model Instance Info",
            json=False,
            keys=[
                "model_name",
                "model_variant_name",
                "name",
                "slug",
                "id",
                "model_version_id",
                "model_id",
                "created",
                "updated",
                "platforms",
                "is_public",
                "yolo_version",
                "model_precision_type",
                "is_nn_archive",
                "downloads",
            ],
            rename={"model_version_id": "variant_id"},
        )
    return ModelInstanceResponse(**data)


@app.command(name="download")
def download_instance(
    identifier: UUID | str,
    output_dir: str | None = None,
    force: bool = False,
) -> Path:
    """Downloads files from a model instance.

    Parameters
    ----------
    identifier : UUID | str
        The model instance ID or slug.
    output_dir : str | None
        The directory to save the downloaded files.
        If not specified, the files will be saved in the current directory.
    force : bool
        Whether to force download the files even if they already exist.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    dest = Path(output_dir) if output_dir else None
    model_instance_id = get_resource_id(identifier, "modelInstances")
    downloaded_path = None
    urls = Request.get(
        service="models",
        endpoint=f"modelInstances/{model_instance_id}/download",
    )
    if not urls:
        raise ValueError("No files to download")

    def cleanup(sigint: int, _: FrameType | None) -> None:
        nonlocal file_path
        logger.info(f"Received signal {sigint}. Download interrupted...")
        file_path.unlink(missing_ok=True)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for url in urls:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            filename = unquote(Path(urlparse(url).path).name)
            if dest is None:
                dest = Path(
                    Request.get(
                        service="models",
                        endpoint=f"modelInstances/{model_instance_id}",
                    ).get("slug", model_instance_id)
                )
            dest.mkdir(parents=True, exist_ok=True)

            file_path = dest / filename
            if file_path.exists() and not force:
                logger.info(
                    f"File '{filename}' already exists. Skipping download. "
                    "Use `force=True` to overwrite."
                )
                downloaded_path = file_path
                continue

            try:
                with open(file_path, "wb") as f, Progress() as progress:
                    task = progress.add_task(
                        f"Downloading '{filename}'", total=total_size
                    )
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            except:
                logger.error(f"Failed to download '{filename}'")
                file_path.unlink(missing_ok=True)
                raise

            logger.info(f"Downloaded '{file_path.name}'")
            downloaded_path = file_path

    assert downloaded_path is not None

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.download", properties={"instance_id": identifier}, include_system_metadata=True)

    return downloaded_path


@overload
def create_instance(
    name: str,
    *,
    variant_id: UUID | str,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_precision_type: TargetPrecision | None = None,
    quantization_data: Quantization | str | None = None,
    tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    yolo_version: YoloVersion | None = None,
    silent: bool = True,
) -> ModelInstanceResponse:
    ...

@overload
def create_instance(
    name: str,
    *,
    variant_id: UUID | str,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_precision_type: TargetPrecision | None = None,
    quantization_data: Quantization | str | None = None,
    tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    yolo_version: YoloVersion | None = None,
    silent: bool = False,
) -> None:
    ...

@overload
def create_instance(
    name: str,
    *,
    variant_id: UUID | str,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_precision_type: TargetPrecision | None = None,
    quantization_data: Quantization | str | None = None,
    tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    yolo_version: YoloVersion | None = None,
    silent: bool | None = None,
) -> ModelInstanceResponse:
    ...

@app.command(name="create")
def create_instance(
    name: str,
    *,
    variant_id: UUID | str,
    model_type: ModelType | None = None,
    parent_id: UUID | str | None = None,
    model_precision_type: TargetPrecision | None = None,
    quantization_data: Quantization | str | None = None,
    tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    yolo_version: YoloVersion | None = None,
    silent: bool | None = None,
) -> ModelInstanceResponse | None:
    """Creates a new model instance.

    Parameters
    ----------
    name : str
        The name of the model instance.
    variant_id : UUID | str
        The ID of the model variant to create an instance for.
    model_type : ModelType | None
        The type of the model.
    parent_id : UUID | str | None
        The ID of the parent model instance.
    model_precision_type : TargetPrecision | None
        The precision type of the model.
    quantization_data : Quantization | None
        The quantization data for the model. Can be one of
        predefined datasets or a dataset id.
    tags : list[str] | None
        List of tags for the model instance.
    input_shape : list[int] | None
        The input shape of the model instance.
    is_deployable : bool | None
        Whether the model instance is deployable.
    yolo_version: YoloVersion | None
        The YOLO version of the model instance if it is a YOLO model.
    silent : bool
        Whether to print the model instance information after creation.
    """

    if silent is None:
        silent = not is_cli_call()
    data = {
        "name": name,
        "model_version_id": str(variant_id) if variant_id else None,
        "parent_id": str(parent_id) if parent_id else None,
        "model_type": model_type,
        "model_precision_type": model_precision_type,
        "tags": tags or [],
        "input_shape": [input_shape] if input_shape else None,
        "quantization_data": quantization_data,
        "is_deployable": is_deployable,
        "yolo_version": yolo_version,
    }
    res = Request.post(service="models", endpoint="modelInstances", json=data)
    logger.info(
        f"Model instance '{res['name']}' created with ID '{res['id']}'"
    )

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.create", properties=data, include_system_metadata=True)

    return get_instance(res["id"])


@app.command(name="delete")
def delete_instance(identifier: UUID | str) -> None:
    """Deletes a model instance.

    Parameters
    ----------
    identifier : UUID | str
        The model instance ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    instance_id = get_resource_id(identifier, "modelInstances")
    Request.delete(service="models", endpoint=f"modelInstances/{instance_id}")
    logger.info(f"Model instance '{identifier}' deleted")

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.delete", properties={"instance_id": identifier}, include_system_metadata=True)


@overload
def get_config(identifier: UUID | str) -> ArchiveConfigurationResponse:
    ...

@overload
def get_config(identifier: UUID | str) -> None:
    ...

@app.command(name="config")
def get_config(identifier: UUID | str) -> ArchiveConfigurationResponse | None:
    """Returns the configuration of a model instance.

    Parameters
    ----------
    identifier : UUID | str
        The model instance ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    silent = not is_cli_call()
    model_instance_id = get_resource_id(identifier, "modelInstances")
    data = Request.get(
        service="models", endpoint=f"modelInstances/{model_instance_id}/config"
    )

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.config", properties={"instance_id": identifier}, include_system_metadata=True)

    if not silent:
        logger.info(data)
        return None
    return ArchiveConfigurationResponse(**data)


@overload
def get_files(identifier: UUID | str) -> list[ModelInstanceFileResponse]:
    ...

@overload
def get_files(identifier: UUID | str) -> None:
    ...

@app.command(name="files")
def get_files(
    identifier: UUID | str,
) -> list[ModelInstanceFileResponse] | None:
    """Returns the files of a model instance.

    Parameters
    ----------
    identifier : UUID | str
        The model instance ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    silent = not is_cli_call()
    model_instance_id = get_resource_id(identifier, "modelInstances")
    data = Request.get(
        service="models", endpoint=f"modelInstances/{model_instance_id}/files"
    )

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.files", properties={"instance_id": identifier}, include_system_metadata=True)

    if not silent:
        logger.info(data)
        return None
    return [ModelInstanceFileResponse(**file) for file in data]


@app.command(name="upload")
def upload_file(file_path: str, identifier: UUID | str) -> None:
    """Uploads a file to a model instance.

    Parameters
    ----------
    file_path : str
        The path to the file to upload.
    identifier : UUID | str
        The model instance ID or slug.
    """
    if isinstance(identifier, UUID):
        identifier = str(identifier)
    model_instance_id = get_resource_id(identifier, "modelInstances")
    with open(file_path, "rb") as file:
        files = {"files": file}
        Request.post(
            service="models",
            endpoint=f"modelInstances/{model_instance_id}/upload",
            files=files,
        )
    logger.info(
        f"File '{file_path}' uploaded to model instance '{identifier}'"
    )

    telemetry = get_telemetry()
    if telemetry:
        telemetry.capture("instances.upload", properties={"instance_id": identifier}, include_system_metadata=True)
