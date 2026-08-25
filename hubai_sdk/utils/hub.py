import shutil
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Literal, NoReturn, TypeVar
from uuid import UUID

from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.typing import Params
from packaging.version import Version
from requests.exceptions import HTTPError
from rich import print
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from hubai_sdk.errors import (
    HubApiError,
    InputError,
    ResourceAmbiguousError,
    ResourceConflictError,
    ResourceNotFoundError,
    ValidationError,
)
from hubai_sdk.utils.config import Config, SingleStageConfig
from hubai_sdk.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from hubai_sdk.utils.filesystem_utils import resolve_path
from hubai_sdk.utils.general import sanitize_net_name
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import EnumJobStatusType
from hubai_sdk.utils.nn_archive import process_nn_archive
from hubai_sdk.utils.sdk_models import JobMessageResponse
from hubai_sdk.utils.types import ModelType, Target

ResourceEndpoint = Literal["models", "modelVersions", "modelInstances"]
T = TypeVar("T")


def get_output_dir_name(
    target: Target, name: str, output_dir: str | None
) -> Path:
    """Build an output directory path for a conversion result."""
    name = sanitize_net_name(name)
    date = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    if output_dir is not None:
        if (OUTPUTS_DIR / output_dir).exists():
            shutil.rmtree(OUTPUTS_DIR / output_dir)
    else:
        output_dir = f"{name}_to_{target.name.lower()}_{date}"
    return OUTPUTS_DIR / output_dir


def init_dirs() -> None:
    """Create the SDK working directories if they do not exist."""
    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)


def get_configs(
    path: str | None, opts: list[str] | dict[str, Any] | None = None
) -> tuple[Config, NNArchiveConfig | None, str | None]:
    """Sets up the configuration.

    Args:
        path: Path to the configuration file or NN Archive.
        opts: Optional CLI overrides for the config file.

    Returns:
        The parsed modelconverter configuration, optional NN Archive
        configuration, and main stage key.
    """

    opts = opts or []
    if isinstance(opts, list):
        if len(opts) % 2 != 0:
            raise InputError(
                "Invalid number of overrides. See --help for more information."
            )
        overrides: Params = {
            opts[i]: opts[i + 1] for i in range(0, len(opts), 2)
        }
    else:
        overrides = opts
    if path is not None:
        path_ = resolve_path(path, MISC_DIR)
        if path_.is_dir() or is_nn_archive(path_):
            logger.info(f"Processing NN archive: {path_}")
            return process_nn_archive(path_, overrides)
    cfg = Config.get_config(path, overrides)

    main_stage_key = None
    if len(cfg.stages) > 1:
        for key in cfg.stages:
            if "yolov8" in key and "seg" in key:
                logger.info(f"Detected main stage key: {key}")
                main_stage_key = key
                break
    else:
        main_stage_key = next(iter(cfg.stages.keys()))

    return cfg, None, main_stage_key


def print_hub_resource_info(
    model: dict[str, Any],
    keys: list[str],
    json: bool,
    rename: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Render a single HubAI resource in CLI-friendly rich output."""
    rename = rename or {}

    if json:
        print(model)
        return

    console = Console()

    if model.get("description_short"):
        description_short_panel = Panel(
            f"[italic]{model['description_short']}[/italic]",
            border_style="dim",
            box=ROUNDED,
        )
    else:
        description_short_panel = None

    table = Table(show_header=False, box=None)
    table.add_column(justify="right", style="bold")
    table.add_column()
    for key in keys:
        if key in ["created", "updated", "last_version_added"]:
            value = model.get(key, "N/A")

            def format_date(date_str: str) -> RenderableType:
                try:
                    date_obj = datetime.strptime(
                        date_str, "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    return date_obj.strftime("%B %d, %Y %H:%M:%S")
                except (ValueError, TypeError):
                    return Pretty("N/A")

            formatted_value = format_date(value)

            table.add_row(
                f"{rename.get(key, key).replace('_', ' ').title()}:",
                formatted_value,
            )
        elif key == "is_public":
            if key not in model:
                value = "N/A"
            else:
                value = "Public" if model[key] else "Private"
            table.add_row("Visibility", Pretty(value))
        elif key == "is_commercial":
            # TODO: Is Usage the right term?
            if key not in model:
                value = "N/A"
            else:
                value = "Commercial" if model[key] else "Non-Commercial"
            table.add_row("Usage:", Pretty(value))
        elif key == "is_nn_archive":
            table.add_row("NN Archive:", Pretty(model.get(key, False)))
        else:
            table.add_row(
                f"{rename.get(key, key).replace('_', ' ').title()}:",
                Pretty(model.get(key, "N/A")),
            )

    info_panel = Panel(table, border_style="cyan", box=ROUNDED, **kwargs)

    if model.get("description"):
        description_panel = Panel(
            Markdown(model["description"]),
            title="Description",
            border_style="green",
            box=ROUNDED,
        )
    else:
        description_panel = None

    nested_panels = []
    if description_short_panel:
        nested_panels.append(description_short_panel)
    nested_panels.append(info_panel)
    if description_panel:
        nested_panels.append(description_panel)

    content = Group(*nested_panels)

    main_panel = Panel(
        content,
        title=f"[bold magenta]{model.get('name', 'N/A')}[/bold magenta]",
        width=74,
        border_style="magenta",
        box=ROUNDED,
    )

    console.print(main_panel)


def print_hub_ls(
    data: list[dict[str, Any]],
    keys: list[str],
    rename: dict[str, str] | None = None,
) -> None:
    """Render a list of HubAI resources as a rich table."""
    rename = rename or {}
    table = Table(row_styles=["yellow", "cyan"], box=ROUNDED)
    for key in keys:
        table.add_column(rename.get(key, key), header_style="magenta i")

    for model in data:
        renderables = []
        for key in keys:
            value = str(model.get(key, "N/A"))
            if isinstance(value, list):
                value = ", ".join(value)
            renderables.append(value)
        table.add_row(*renderables)

    console = Console()
    console.print(table)


def is_valid_uuid(uuid_string: str) -> bool:
    """Return whether a string is a valid UUID."""
    try:
        UUID(uuid_string)
    except Exception:
        return False
    else:
        return True


def is_hubai_id(identifier: str) -> bool:
    """Return whether an identifier looks like a HubAI resource ID."""
    temp = identifier
    if not temp.startswith("ai"):
        return False

    return "_" in identifier


def slug_to_id(
    slug: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> str | None:
    """Resolve a raw resource slug to its resource ID."""
    matches: dict[str, dict[str, Any]] = {}
    for is_public in [True, False]:
        params = {
            "is_public": is_public,
            "slug": slug,
        }
        try:
            data = Request.get(
                service="models", endpoint=f"{endpoint}/", params=params
            )
        except HTTPError as exc:
            raise_for_hub_error(exc, identifier=slug, endpoint=endpoint)
        if data:
            for item in data:
                if item["slug"] == slug:
                    matches[str(item["id"])] = item

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ResourceAmbiguousError(slug, endpoint)

    return None


def get_resource_id(
    identifier: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> str:
    """Resolve a UUID, HubAI ID, raw slug, or resource path to a resource ID.

    Args:
        identifier: User-supplied UUID, HubAI ID, raw slug, or resource path.
        endpoint: Resource endpoint to resolve against.

    Returns:
        The resolved HubAI resource ID.

    Raises:
        ResourceAmbiguousError: If the identifier matches multiple resources.
        ValueError: If the identifier cannot be resolved.
    """
    if is_valid_uuid(identifier):
        return identifier

    if is_hubai_id(identifier):
        return identifier

    resource_path_id = _resource_path_to_id(identifier, endpoint)
    if resource_path_id:
        return resource_path_id

    found_id = slug_to_id(identifier, endpoint)

    if not found_id:
        raise ValueError(
            f"Resource for endpoint '{endpoint}' with identifier '{identifier}' not found in HubAI."
        )

    return found_id


def _resource_path_to_id(
    identifier: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> str | None:
    """Resolve a human-readable HubAI resource path to its resource ID.

    Supported forms are ``<team>/<model>`` for models;
    ``<model>:<variant>`` and ``<model>:<variant>:<version>`` for model
    versions; and ``<model>:<variant>:<instance-hash>`` for model instances.
    Model-version and instance paths may include an optional ``<team>/``
    prefix. The team
    segment, when present, is matched against the model response's
    ``team_slug`` field.
    """
    if endpoint == "models":
        parsed_model = _parse_model_resource_path(identifier)
        if parsed_model is None:
            return None
        team_slug, model_slug = parsed_model
        return _find_model_id(model_slug, team_slug, identifier)

    if endpoint == "modelVersions":
        parsed_variant = _parse_variant_resource_path(identifier)
        version = None
        if parsed_variant is None:
            parsed_versioned_variant = _parse_versioned_variant_resource_path(
                identifier
            )
            if parsed_versioned_variant is None:
                return None
            team_slug, model_slug, variant_slug, version = (
                parsed_versioned_variant
            )
        else:
            team_slug, model_slug, variant_slug = parsed_variant
        model_identifier = (
            f"{team_slug}/{model_slug}" if team_slug else model_slug
        )
        model_id = get_resource_id(model_identifier, "models")
        return _find_variant_id(model_id, variant_slug, identifier, version)

    if endpoint == "modelInstances":
        parsed_instance = _parse_instance_resource_path(identifier)
        if parsed_instance is None:
            return None
        variant_path, instance_hash = parsed_instance
        variant_id = get_resource_id(variant_path, "modelVersions")
        return _find_instance_id(variant_id, instance_hash, identifier)

    return None


def _parse_model_resource_path(identifier: str) -> tuple[str, str] | None:
    """Parse a ``<team>/<model>`` resource path."""
    if identifier.count("/") != 1 or ":" in identifier:
        return None
    team_slug, model_slug = identifier.split("/", maxsplit=1)
    if not team_slug or not model_slug:
        return None
    return team_slug, model_slug


def _parse_variant_resource_path(
    identifier: str,
) -> tuple[str | None, str, str] | None:
    """Parse a ``model:variant`` resource path with an optional team segment."""
    if identifier.count(":") != 1:
        return None
    model_identifier, variant_slug = identifier.split(":", maxsplit=1)
    if not model_identifier or not variant_slug:
        return None

    if "/" not in model_identifier:
        return None, model_identifier, variant_slug

    parsed_model = _parse_model_resource_path(model_identifier)
    if parsed_model is None:
        return None
    team_slug, model_slug = parsed_model
    return team_slug, model_slug, variant_slug


def _parse_instance_resource_path(identifier: str) -> tuple[str, str] | None:
    """Parse a ``model:variant:instance-hash`` resource path with an
    optional team segment."""
    if identifier.count(":") != 2:
        return None
    variant_path, instance_hash = identifier.rsplit(":", maxsplit=1)
    if not instance_hash or _parse_variant_resource_path(variant_path) is None:
        return None
    return variant_path, instance_hash


def _parse_versioned_variant_resource_path(
    identifier: str,
) -> tuple[str | None, str, str, str] | None:
    """Parse a ``model:variant:version`` resource path with an optional team
    segment."""
    if identifier.count(":") != 2:
        return None
    variant_path, version = identifier.rsplit(":", maxsplit=1)
    parsed_variant = _parse_variant_resource_path(variant_path)
    if not version or parsed_variant is None:
        return None
    team_slug, model_slug, variant_slug = parsed_variant
    return team_slug, model_slug, variant_slug, version


def _find_model_id(
    model_slug: str, team_slug: str | None, identifier: str
) -> str | None:
    """Find one model by slug, optionally constraining its owning team."""
    matches: dict[str, dict[str, Any]] = {}
    for is_public in [True, False]:
        try:
            data = Request.get(
                service="models",
                endpoint="models/",
                params={
                    "is_public": is_public,
                    "slug": model_slug,
                    "limit": 500,
                },
            )
        except HTTPError as exc:
            raise_for_hub_error(exc, identifier=identifier, endpoint="models")
        for item in data:
            if item["slug"] != model_slug:
                continue
            if team_slug is not None and item.get("team_slug") != team_slug:
                continue
            matches[str(item["id"])] = item

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ResourceAmbiguousError(identifier, "models")
    return None


def _find_variant_id(
    model_id: str,
    variant_slug: str,
    identifier: str,
    version: str | None = None,
) -> str | None:
    """Find one model version beneath a resolved model."""
    matches: dict[str, dict[str, Any]] = {}
    for is_public in [True, False]:
        try:
            data = Request.get(
                service="models",
                endpoint="modelVersions",
                params={
                    "model_id": model_id,
                    "variant_slug": variant_slug,
                    "version": version,
                    "is_public": is_public,
                    "limit": 500,
                },
            )
        except HTTPError as exc:
            raise_for_hub_error(
                exc, identifier=identifier, endpoint="modelVersions"
            )
        for item in data:
            if item.get("variant_slug") != variant_slug:
                continue
            if version is not None and item.get("version") != version:
                continue
            matches[str(item["id"])] = item

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ResourceAmbiguousError(identifier, "modelVersions")
    return None


def _find_instance_id(
    variant_id: str, instance_hash: str, identifier: str
) -> str | None:
    """Find one instance beneath a resolved variant by its short hash."""
    matches: dict[str, dict[str, Any]] = {}
    for is_public in [True, False]:
        try:
            data = Request.get(
                service="models",
                endpoint="modelInstances",
                params={
                    "model_version_id": variant_id,
                    "is_public": is_public,
                    "limit": 500,
                },
            )
        except HTTPError as exc:
            raise_for_hub_error(
                exc, identifier=identifier, endpoint="modelInstances"
            )
        for item in data:
            if item.get("hash_short") == instance_hash:
                matches[str(item["id"])] = item

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ResourceAmbiguousError(identifier, "modelInstances")
    return None


def resolve_resource_id(
    identifier: str,
    endpoint: ResourceEndpoint,
) -> str:
    """Resolve an identifier and translate lookup failures to SDK
    errors."""
    try:
        return get_resource_id(identifier, endpoint)
    except ValueError as exc:
        raise ResourceNotFoundError(identifier, endpoint) from exc
    except HTTPError as exc:
        raise_for_hub_error(exc)


def get_resource_info(
    identifier: str,
    endpoint: ResourceEndpoint,
) -> dict[str, Any]:
    """Fetch raw HubAI resource data for a resolved identifier."""
    resource_id = resolve_resource_id(identifier, endpoint)

    try:
        return Request.get(
            service="models", endpoint=f"{endpoint}/{resource_id}/"
        )
    except HTTPError as exc:
        raise_for_hub_error(exc, identifier=identifier, endpoint=endpoint)


def raise_for_hub_error(
    exc: HTTPError,
    *,
    identifier: str | None = None,
    endpoint: ResourceEndpoint | Literal["jobs"] | None = None,
    conflict_message: str | None = None,
) -> NoReturn:
    """Translate a requests HTTP error into an SDK-specific exception."""
    status_code = (
        exc.response.status_code if exc.response is not None else None
    )
    detail = _get_http_error_detail(exc)
    default_message = (
        f"HubAI request failed with status code {status_code}."
        if status_code is not None
        else "HubAI request failed."
    )

    if status_code == 404 and identifier is not None and endpoint is not None:
        raise ResourceNotFoundError(identifier, endpoint) from exc

    # The backend currently reports duplicate model/variant creates as 422
    # with "Unique constraint error" in detail, but we still accept 409.
    if (
        detail is not None and "Unique constraint error" in detail
    ) or status_code == 409:
        raise ResourceConflictError(
            conflict_message or detail or "Resource already exists in HubAI.",
            status_code=status_code,
        ) from exc

    if status_code in {400, 422}:
        raise ValidationError(
            detail or default_message,
            status_code=status_code,
        ) from exc

    raise HubApiError(
        detail or default_message,
        status_code=status_code,
    ) from exc


def run_cli(action: Callable[[], T]) -> T:
    """Run a CLI action and convert SDK exceptions into clean exits."""
    try:
        return action()
    except HubApiError as exc:
        from hubai_sdk.utils.telemetry import record_cli_failure_reason

        record_cli_failure_reason(exc)
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from None
    except (InputError, FileNotFoundError) as exc:
        from hubai_sdk.utils.telemetry import record_cli_failure_reason

        record_cli_failure_reason(exc)
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from None
    except HTTPError as exc:
        from hubai_sdk.utils.telemetry import record_cli_failure_reason

        record_cli_failure_reason(exc)
        sys.stderr.write(f"{_get_http_error_detail(exc)}\n")
        raise SystemExit(1) from None


def get_variant_name(
    cfg: SingleStageConfig, model_type: ModelType, name: str
) -> str:
    """Derive a variant name, appending input resolution when known."""
    shape = cfg.inputs[0].shape
    layout = cfg.inputs[0].layout

    if shape is not None:
        if layout is not None and "H" in layout and "W" in layout:
            h, w = shape[layout.index("H")], shape[layout.index("W")]
            return f"{name} {h}x{w}"
        if len(shape) == 4:
            if model_type == ModelType.TFLITE:
                h, w = shape[1], shape[2]
            else:
                h, w = shape[2], shape[3]
            return f"{name} {h}x{w}"
    return name


def get_version_number(model_id: str) -> str:
    """Return the next patch version for a model's variants."""
    try:
        versions = Request.get(
            service="models",
            endpoint="modelVersions/",
            params={"model_id": model_id},
        )
    except HTTPError as exc:
        raise_for_hub_error(exc)
    if not versions:
        return "0.1.0"
    max_version = Version(versions[0]["version"])
    for v in versions[1:]:
        max_version = max(max_version, Version(v["version"]))
    max_version = str(max_version)
    version_numbers = max_version.split(".")
    version_numbers[-1] = str(int(version_numbers[-1]) + 1)
    return ".".join(version_numbers)


def wait_for_job(job_id: str) -> JobMessageResponse:
    """Poll a HubAI job until it completes or fails.

    Args:
        job_id: HubAI job ID.

    Returns:
        The completed job response.

    Raises:
        HubApiError: If the job fails or ends in a terminal non-success
            status.
    """

    def _get_job(job_id: str) -> JobMessageResponse:
        try:
            response = Request.get(service="jobs", endpoint=f"jobs/{job_id}")
        except HTTPError as exc:
            raise_for_hub_error(exc, identifier=job_id, endpoint="jobs")
        return JobMessageResponse(**response)

    while True:
        job = _get_job(job_id)
        if job.status == EnumJobStatusType.COMPLETED:
            return job
        if job.status == EnumJobStatusType.FAILED:
            raise HubApiError(
                f"Job '{job_id}' failed with error: {job.exception}"
            )
        if job.status in {
            EnumJobStatusType.CANCELLED,
            EnumJobStatusType.SHUTDOWN,
        }:
            detail = f": {job.exception}" if job.exception else ""
            raise HubApiError(
                f"Job '{job_id}' ended with status {job.status.value}{detail}"
            )
        sleep(1)


def get_target_specific_options(
    target: Target, cfg: SingleStageConfig, tool_version: str | None = None
) -> dict[str, Any]:
    """Build export options for a target from a parsed config."""
    json_cfg = cfg.model_dump(mode="json")
    inputs = [
        {
            "input_name": input_config["name"],
            **{
                key: value
                for key, value in input_config.items()
                if key != "name"
            },
        }
        for input_config in json_cfg["inputs"]
    ]
    options = {
        "disable_onnx_simplification": cfg.disable_onnx_simplification,
        "disable_onnx_optimization": cfg.disable_onnx_optimization,
        "inputs": inputs,
    }
    if target is Target.RVC4:
        options["snpe_onnx_to_dlc_args"] = cfg.rvc4.snpe_onnx_to_dlc_args
        options["snpe_dlc_quant_args"] = cfg.rvc4.snpe_dlc_quant_args
        options["snpe_dlc_graph_prepare_args"] = (
            cfg.rvc4.snpe_dlc_graph_prepare_args
        )
        if tool_version is not None:
            options["snpe_version"] = tool_version
    elif target in [Target.RVC2, Target.RVC3]:
        target_cfg = getattr(cfg, target.value)
        options["mo_args"] = target_cfg.mo_args
        options["compile_tool_args"] = target_cfg.compile_tool_args
        if tool_version is not None:
            options["ir_version"] = tool_version
        if target is Target.RVC3:
            options["pot_target_device"] = cfg.rvc3.pot_target_device.value
        if target is Target.RVC2:
            options["superblob"] = cfg.rvc2.superblob
            options["number_of_shaves"] = cfg.rvc2.number_of_shaves
    elif target is Target.HAILO:
        options["optimization_level"] = cfg.hailo.optimization_level
        options["compression_level"] = cfg.hailo.compression_level
        options["batch_size"] = cfg.hailo.batch_size
        options["disable_calibration"] = cfg.hailo.disable_calibration

    return options


def _get_http_error_detail(exc: HTTPError) -> str:
    """Extract a readable error message from a requests HTTPError."""
    status_code = (
        exc.response.status_code if exc.response is not None else None
    )
    response = exc.response
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            if response.text:
                return response.text
        else:
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, list):
                    formatted = _format_http_error_items(detail)
                    if formatted:
                        return formatted
                if detail:
                    return str(detail)
                if payload:
                    return str(payload)
            elif isinstance(payload, list):
                formatted = _format_http_error_items(payload)
                if formatted:
                    return formatted
                if payload:
                    return str(payload)
            elif payload:
                return str(payload)

    message = str(exc)
    if message:
        return message
    if status_code is not None:
        return f"HubAI request failed with status code {status_code}."
    return "HubAI request failed."


def _format_http_error_items(items: list[object]) -> str | None:
    """Format validation-style error items into a single message
    string."""
    messages: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            messages.append(str(item))
            continue

        message = item.get("msg")
        location = item.get("loc")
        if message is None:
            messages.append(str(item))
        elif isinstance(location, list) and location:
            loc_str = ".".join(str(part) for part in location)
            messages.append(f"{loc_str}: {message}")
        else:
            messages.append(str(message))

    if messages:
        return "; ".join(messages)
    return None
