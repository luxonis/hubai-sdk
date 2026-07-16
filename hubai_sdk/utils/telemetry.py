import inspect
import os
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import cyclopts
import requests
from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    TelemetryDefaults,
    get_or_init,
    system_context_provider,
)
from luxonis_ml.telemetry.suppression import is_suppressed

from hubai_sdk.errors import (
    HubApiError,
    InputError,
    ResourceConflictError,
    ResourceNotFoundError,
    ValidationError,
)
from hubai_sdk.utils.types import ModelType

HUBAI_SDK_TELEMETRY_DEFAULTS = TelemetryDefaults(
    enabled=True,
    backend="posthog",
    api_key="phc_ojEByaCiZZ5eigzaM43PaEVbfLfFDF5NgkXEMPabrT9a",
    endpoint="https://us.i.posthog.com",
)

COMMAND_EVENT = "hubai_sdk_command_ran"
CLIENT_INITIALIZED_EVENT = "hubai_sdk_client_initialized"
OPERATION_RESULT_EVENT = "hubai_sdk_operation_result_recorded"
MODELS_LISTED_EVENT = "hubai_sdk_models_listed"
MODEL_RETRIEVED_EVENT = "hubai_sdk_model_retrieved"
MODEL_CREATED_EVENT = "hubai_sdk_model_created"
MODEL_UPDATED_EVENT = "hubai_sdk_model_updated"
MODEL_DELETED_EVENT = "hubai_sdk_model_deleted"
VARIANTS_LISTED_EVENT = "hubai_sdk_variants_listed"
VARIANT_RETRIEVED_EVENT = "hubai_sdk_variant_retrieved"
VARIANT_CREATED_EVENT = "hubai_sdk_variant_created"
VARIANT_DELETED_EVENT = "hubai_sdk_variant_deleted"
INSTANCES_LISTED_EVENT = "hubai_sdk_instances_listed"
INSTANCE_RETRIEVED_EVENT = "hubai_sdk_instance_retrieved"
INSTANCE_CREATED_EVENT = "hubai_sdk_instance_created"
INSTANCE_DELETED_EVENT = "hubai_sdk_instance_deleted"
INSTANCE_CONFIG_RETRIEVED_EVENT = "hubai_sdk_instance_config_retrieved"
INSTANCE_FILES_LISTED_EVENT = "hubai_sdk_instance_files_listed"
INSTANCE_DOWNLOAD_COMPLETED_EVENT = "hubai_sdk_instance_download_completed"
INSTANCE_FILE_UPLOADED_EVENT = "hubai_sdk_instance_file_uploaded"
CONVERSION_CONFIGURED_EVENT = "hubai_sdk_conversion_configured"
CONVERSION_RESULT_EVENT = "hubai_sdk_conversion_result_recorded"
CONVERSION_FLOW_NAME = "hubai_sdk_conversion_lifecycle"

_CLI_FAILURE_REASON: ContextVar["FailureReason | None"] = ContextVar(
    "hubai_sdk_cli_failure_reason", default=None
)
_CONVERSION_RUN_ID: ContextVar[str | None] = ContextVar(
    "hubai_sdk_conversion_run_id", default=None
)


class TelemetryResult(str, Enum):
    SUCCESS = "success"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class FailureReason(str, Enum):
    USER_INTERRUPT = "user_interrupt"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    AUTH_ERROR = "auth_error"
    SERVER_ERROR = "server_error"
    API_ERROR = "api_error"
    CONFIG_ERROR = "config_error"
    UPLOAD_ERROR = "upload_error"
    EXPORT_ERROR = "export_error"
    DOWNLOAD_ERROR = "download_error"
    UNKNOWN = "unknown"


class InvocationSurface(str, Enum):
    CLI = "cli"
    PYTHON_API = "python_api"


class ApiKeySource(str, Enum):
    ARGUMENT = "argument"
    ENVIRONMENT = "environment"
    STORED_CREDENTIALS = "stored_credentials"


class ConversionPhase(str, Enum):
    CONFIGURATION = "configuration"
    RESOURCE_SETUP = "resource_setup"
    UPLOAD = "upload"
    EXPORT = "export"
    DOWNLOAD = "download"


class ConversionFlowStep(str, Enum):
    CONFIGURATION_RESOLVED = "configuration_resolved"
    RESULT_RECORDED = "result_recorded"


class TelemetryGroup(str, Enum):
    AUTH = "auth"
    CLIENT = "client"
    CONVERSION = "conversion"
    MODELS = "models"
    VARIANTS = "variants"
    INSTANCES = "instances"


class TargetResource(str, Enum):
    MODEL = "model"
    VARIANT = "variant"
    INSTANCE = "instance"
    CONVERSION = "conversion"


class OperationName(str, Enum):
    CLIENT_INITIALIZE = "client_initialize"
    CONVERT = "convert"
    MODELS_LIST = "models_list"
    MODEL_GET = "model_get"
    MODEL_CREATE = "model_create"
    MODEL_UPDATE = "model_update"
    MODEL_DELETE = "model_delete"
    VARIANTS_LIST = "variants_list"
    VARIANT_GET = "variant_get"
    VARIANT_CREATE = "variant_create"
    VARIANT_DELETE = "variant_delete"
    INSTANCES_LIST = "instances_list"
    INSTANCE_GET = "instance_get"
    INSTANCE_DOWNLOAD = "instance_download"
    INSTANCE_CREATE = "instance_create"
    INSTANCE_DELETE = "instance_delete"
    INSTANCE_CONFIG_GET = "instance_config_get"
    INSTANCE_FILES_LIST = "instance_files_list"
    INSTANCE_UPLOAD = "instance_upload"


class CommandName(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    CONVERT = "convert"
    MODEL_LIST = "model_ls"
    MODEL_INFO = "model_info"
    MODEL_CREATE = "model_create"
    MODEL_UPDATE = "model_update"
    MODEL_DELETE = "model_delete"
    VARIANT_LIST = "variant_ls"
    VARIANT_INFO = "variant_info"
    VARIANT_CREATE = "variant_create"
    VARIANT_DELETE = "variant_delete"
    INSTANCE_LIST = "instance_ls"
    INSTANCE_INFO = "instance_info"
    INSTANCE_DOWNLOAD = "instance_download"
    INSTANCE_CREATE = "instance_create"
    INSTANCE_DELETE = "instance_delete"
    INSTANCE_CONFIG = "instance_config"
    INSTANCE_FILES = "instance_files"
    INSTANCE_UPLOAD = "instance_upload"


class ConfigSource(str, Enum):
    NN_ARCHIVE = "nn_archive"
    YAML_CONFIG = "yaml_config"
    DIRECT_MODEL_INPUT = "direct_model_input"


class IdentifierType(str, Enum):
    UUID = "uuid"
    SLUG = "slug"
    UNKNOWN = "unknown"


class VisibilityFilterValue(str, Enum):
    ALL = "all"
    PUBLIC = "public"
    PRIVATE = "private"


class VisibilityValue(str, Enum):
    TEAM = "team"
    PUBLIC = "public"
    PRIVATE = "private"


class QuantizationInputType(str, Enum):
    NONE = "none"
    CUSTOM_ZIP = "custom_zip"
    DATASET_ID = "dataset_id"
    PREDEFINED_DOMAIN = "predefined_domain"


@dataclass(frozen=True)
class CommandMetadata:
    command_name: CommandName
    command_group: TelemetryGroup


@dataclass(frozen=True)
class OperationTelemetrySpec:
    operation_name: OperationName
    operation_group: TelemetryGroup
    success_event: str | None = None
    target_resource: TargetResource | None = None
    identifier_param: str | None = None
    success_builder: Callable[[dict[str, Any], Any], dict[str, Any]] | None = (
        None
    )


def get_component_telemetry() -> Telemetry:
    return get_or_init(
        "hubai_sdk",
        source_component="hubai-sdk",
        library_version=_sdk_version(),
        config=TelemetryConfig.from_environ(
            defaults=HUBAI_SDK_TELEMETRY_DEFAULTS
        ),
        system_context_providers=[system_context_provider],
    )


def invocation_surface() -> InvocationSurface:
    if os.environ.get("HUBAI_CALL_SOURCE", "").upper() == "CLI":
        return InvocationSurface.CLI
    return InvocationSurface.PYTHON_API


def current_conversion_run_id() -> str | None:
    return _CONVERSION_RUN_ID.get()


def get_or_create_conversion_run_id() -> str:
    conversion_run_id = current_conversion_run_id()
    if conversion_run_id:
        return conversion_run_id
    conversion_run_id = str(uuid4())
    _CONVERSION_RUN_ID.set(conversion_run_id)
    return conversion_run_id


def reset_conversion_run_id(previous_value: str | None) -> None:
    _CONVERSION_RUN_ID.set(previous_value)


def record_cli_failure_reason(exc: BaseException) -> None:
    _CLI_FAILURE_REASON.set(failure_reason_from_exception(exc))


def pop_cli_failure_reason() -> FailureReason | None:
    reason = _CLI_FAILURE_REASON.get()
    _CLI_FAILURE_REASON.set(None)
    return reason


def command_result_from_exception(
    exc: BaseException | None,
) -> TelemetryResult:
    if exc is None:
        return TelemetryResult.SUCCESS
    code = getattr(exc, "code", None)
    if isinstance(exc, SystemExit) and code in {None, 0}:
        return TelemetryResult.SUCCESS
    if isinstance(exc, KeyboardInterrupt):
        return TelemetryResult.INTERRUPTED
    if isinstance(exc, SystemExit) and code == 130:
        return TelemetryResult.INTERRUPTED
    return TelemetryResult.FAILED


def failure_reason_from_exception(
    exc: BaseException | None,
) -> FailureReason | None:
    if exc is None:
        return None
    code = getattr(exc, "code", None)
    if isinstance(exc, KeyboardInterrupt):
        return FailureReason.USER_INTERRUPT
    if isinstance(exc, SystemExit) and code == 130:
        return FailureReason.USER_INTERRUPT
    if isinstance(exc, ResourceNotFoundError):
        return FailureReason.NOT_FOUND
    if isinstance(exc, ResourceConflictError):
        return FailureReason.CONFLICT
    if isinstance(exc, (ValidationError, InputError)):
        return FailureReason.VALIDATION_ERROR
    if isinstance(exc, requests.Timeout):
        return FailureReason.TIMEOUT
    if isinstance(exc, requests.ConnectionError):
        return FailureReason.NETWORK_ERROR
    if isinstance(exc, requests.HTTPError):
        return _failure_reason_from_status_code(
            exc.response.status_code if exc.response is not None else None
        )
    if isinstance(exc, HubApiError):
        if "Invalid API key" in str(exc) or "API key" in str(exc):
            return FailureReason.AUTH_ERROR
        return _failure_reason_from_status_code(exc.status_code)
    if isinstance(exc, ValueError) and "API key" in str(exc):
        return FailureReason.AUTH_ERROR
    return FailureReason.UNKNOWN


def conversion_failure_reason(
    exc: BaseException | None, *, phase: ConversionPhase
) -> FailureReason | None:
    if exc is None:
        return None
    if failure_reason_from_exception(exc) == FailureReason.USER_INTERRUPT:
        return FailureReason.USER_INTERRUPT
    if phase in {
        ConversionPhase.CONFIGURATION,
        ConversionPhase.RESOURCE_SETUP,
    }:
        return FailureReason.CONFIG_ERROR
    if phase == ConversionPhase.UPLOAD:
        return FailureReason.UPLOAD_ERROR
    if phase == ConversionPhase.EXPORT:
        return FailureReason.EXPORT_ERROR
    if phase == ConversionPhase.DOWNLOAD:
        return FailureReason.DOWNLOAD_ERROR
    return FailureReason.UNKNOWN


def telemetry_operation(
    spec: OperationTelemetrySpec,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_suppressed():
                return func(*args, **kwargs)

            telemetry = get_component_telemetry()
            bound = _bind_arguments(signature, args, kwargs)
            start = time.monotonic()
            caught_exc: BaseException | None = None
            result: Any = None

            try:
                # Save the return value so the finally block can inspect it.
                result = func(*args, **kwargs)
                return result  # noqa: RET504, TRY300
            except BaseException as exc:
                caught_exc = exc
                raise
            finally:
                if caught_exc is None and spec.success_event:
                    _capture_success_event(
                        telemetry,
                        spec=spec,
                        bound_arguments=bound,
                        result=result,
                    )
                _capture_operation_result(
                    telemetry,
                    spec=spec,
                    bound_arguments=bound,
                    exc=caught_exc,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        return wrapper

    return decorator


def instrument_hubai_cli(app: cyclopts.App) -> None:
    _wrap_cyclopts(app, prefix="")


def build_command_properties(
    *,
    command_name: CommandName,
    command_group: TelemetryGroup,
    result: TelemetryResult,
    duration_ms: int,
    failure_reason: FailureReason | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "command_name": command_name.value,
            "command_group": command_group.value,
            "result": result.value,
            "failure_reason": (
                failure_reason.value if failure_reason is not None else None
            ),
            "duration_ms": duration_ms,
        }
    )


def build_client_initialized_properties(
    api_key_source: ApiKeySource,
) -> dict[str, Any]:
    return {
        "api_key_source": api_key_source.value,
    }


def build_conversion_flow_properties(
    conversion_run_id: str,
    flow_step: ConversionFlowStep,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "flow_name": CONVERSION_FLOW_NAME,
        "conversion_run_id": conversion_run_id,
        "flow_step": flow_step.value,
        **properties,
    }


def build_conversion_summary(
    *,
    target: str,
    config_source: ConfigSource,
    input_model_type: ModelType,
    invocation_surface: InvocationSurface,
    existing_model_reused: bool,
    existing_variant_reused: bool,
    quantization_mode: str | None,
    quantization_input_type: QuantizationInputType,
    max_quantization_images: int | None,
    yolo_version: str | None,
    yolo_class_names: list[str] | None,
    yolo_input_shape_provided: bool,
    tool_version_provided: bool,
    input_shape_provided: bool,
    download_output_dir_provided: bool,
    input_count: int,
    target_options: Mapping[str, Any],
) -> dict[str, Any]:
    properties = {
        "target": target,
        "config_source": config_source.value,
        "input_model_type": input_model_type.value.lower(),
        "invocation_surface": invocation_surface.value,
        "existing_model_reused": existing_model_reused,
        "existing_variant_reused": existing_variant_reused,
        "quantization_mode": quantization_mode,
        "quantization_input_type": quantization_input_type.value,
        "max_quantization_images_bucket": (
            bucket_max_quantization_images(max_quantization_images)
            if max_quantization_images is not None
            else None
        ),
        "yolo_version": yolo_version,
        "yolo_class_count_bucket": (
            bucket_yolo_class_count(len(yolo_class_names))
            if yolo_class_names
            else None
        ),
        "yolo_input_shape_provided": yolo_input_shape_provided,
        "tool_version_provided": tool_version_provided,
        "input_shape_provided": input_shape_provided,
        "download_output_dir_provided": download_output_dir_provided,
        "disable_onnx_simplification": target_options.get(
            "disable_onnx_simplification"
        ),
        "disable_onnx_optimization": target_options.get(
            "disable_onnx_optimization"
        ),
        "input_count_bucket": bucket_nonzero_count(input_count),
        "rvc2_superblob": target_options.get("superblob"),
        "rvc2_number_of_shaves_bucket": (
            bucket_rvc2_shaves(int(target_options["number_of_shaves"]))
            if target_options.get("number_of_shaves") is not None
            else None
        ),
        "rvc3_pot_target_device": target_options.get("pot_target_device"),
        "hailo_optimization_level": target_options.get("optimization_level"),
        "hailo_compression_level": target_options.get("compression_level"),
        "hailo_batch_size_bucket": (
            bucket_nonzero_count(int(target_options["batch_size"]))
            if target_options.get("batch_size") is not None
            else None
        ),
        "hailo_disable_calibration": target_options.get("disable_calibration"),
        "has_snpe_onnx_to_dlc_args": bool(
            target_options.get("snpe_onnx_to_dlc_args")
        )
        if "snpe_onnx_to_dlc_args" in target_options
        else None,
        "has_snpe_dlc_quant_args": bool(
            target_options.get("snpe_dlc_quant_args")
        )
        if "snpe_dlc_quant_args" in target_options
        else None,
        "has_snpe_dlc_graph_prepare_args": bool(
            target_options.get("snpe_dlc_graph_prepare_args")
        )
        if "snpe_dlc_graph_prepare_args" in target_options
        else None,
    }
    return _drop_none(properties)


def build_conversion_result_properties(
    *,
    result: TelemetryResult,
    duration_ms: int,
    failure_reason: FailureReason | None = None,
    downloaded_file_count: int | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "result": result.value,
            "failure_reason": (
                failure_reason.value if failure_reason is not None else None
            ),
            "duration_ms": duration_ms,
            "downloaded_file_count_bucket": (
                bucket_nonzero_count(downloaded_file_count)
                if downloaded_file_count is not None
                else None
            ),
        }
    )


def config_source_from_path(
    path: str, *, is_archive: bool, is_yaml: bool
) -> ConfigSource:
    if is_archive:
        return ConfigSource.NN_ARCHIVE
    if is_yaml:
        return ConfigSource.YAML_CONFIG
    return ConfigSource.DIRECT_MODEL_INPUT


def identifier_type(identifier: UUID | str | None) -> IdentifierType | None:
    if identifier is None:
        return None
    if isinstance(identifier, UUID):
        return IdentifierType.UUID
    try:
        UUID(str(identifier))
    except (TypeError, ValueError, AttributeError):
        return (
            IdentifierType.SLUG if str(identifier) else IdentifierType.UNKNOWN
        )
    return IdentifierType.UUID


def visibility_filter(is_public: bool | None) -> VisibilityFilterValue:
    if is_public is None:
        return VisibilityFilterValue.ALL
    return (
        VisibilityFilterValue.PUBLIC
        if is_public
        else VisibilityFilterValue.PRIVATE
    )


def visibility_value(is_public: bool | None) -> VisibilityValue:
    if is_public is None:
        return VisibilityValue.TEAM
    return VisibilityValue.PUBLIC if is_public else VisibilityValue.PRIVATE


def quantization_input_type(
    quantization_data: str | None, *, custom_zip: bool = False
) -> QuantizationInputType:
    if quantization_data is None:
        return QuantizationInputType.NONE
    if custom_zip or quantization_data == "CUSTOM":
        return QuantizationInputType.CUSTOM_ZIP
    if quantization_data.startswith("aid_"):
        return QuantizationInputType.DATASET_ID
    return QuantizationInputType.PREDEFINED_DOMAIN


def sort_mode(value: str, *, allowed: set[str]) -> str:
    return value if value in allowed else "custom"


def file_extension(path: str | Path) -> str:
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if suffixes[-2:] == [".tar", ".xz"]:
        return ".tar.xz"
    if suffixes[-2:] == [".tar", ".gz"]:
        return ".tar.gz"
    if suffixes:
        return suffixes[-1]
    return "<none>"


def bucket_zero_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2_4"
    return "5_plus"


def bucket_nonzero_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 4:
        return "2_4"
    return "5_plus"


def bucket_result_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 10:
        return "1_10"
    if value <= 50:
        return "11_50"
    return "51_plus"


def bucket_limit(value: int) -> str:
    if value <= 10:
        return "1_10"
    if value <= 50:
        return "11_50"
    if value <= 100:
        return "51_100"
    return "101_plus"


def bucket_max_quantization_images(value: int) -> str:
    if value <= 32:
        return "1_32"
    if value <= 128:
        return "33_128"
    if value <= 512:
        return "129_512"
    return "513_plus"


def bucket_yolo_class_count(value: int) -> str:
    if value <= 20:
        return "1_20"
    if value <= 80:
        return "21_80"
    return "81_plus"


def bucket_rvc2_shaves(value: int) -> str:
    if value <= 4:
        return "1_4"
    if value <= 8:
        return "5_8"
    return "9_plus"


def bucket_file_size(value: int) -> str:
    megabyte = 1024 * 1024
    gigabyte = 1024 * megabyte
    if value < 10 * megabyte:
        return "under_10m"
    if value < 100 * megabyte:
        return "10m_100m"
    if value < gigabyte:
        return "100m_1g"
    return "above_1g"


def capture_client_initialized(api_key_source: ApiKeySource) -> None:
    get_component_telemetry().capture(
        CLIENT_INITIALIZED_EVENT,
        build_client_initialized_properties(api_key_source),
        include_system_metadata=True,
    )


def capture_conversion_configured(
    conversion_run_id: str, properties: dict[str, Any]
) -> None:
    get_component_telemetry().capture(
        CONVERSION_CONFIGURED_EVENT,
        build_conversion_flow_properties(
            conversion_run_id,
            ConversionFlowStep.CONFIGURATION_RESOLVED,
            properties,
        ),
        include_system_metadata=True,
        distinct_id=conversion_run_id,
    )


def capture_conversion_result(
    conversion_run_id: str, properties: dict[str, Any]
) -> None:
    get_component_telemetry().capture(
        CONVERSION_RESULT_EVENT,
        build_conversion_flow_properties(
            conversion_run_id,
            ConversionFlowStep.RESULT_RECORDED,
            properties,
        ),
        include_system_metadata=True,
        distinct_id=conversion_run_id,
    )


def capture_operation_result(
    *,
    spec: OperationTelemetrySpec,
    exc: BaseException | None,
    duration_ms: int,
    bound_arguments: dict[str, Any] | None = None,
) -> None:
    _capture_operation_result(
        get_component_telemetry(),
        spec=spec,
        bound_arguments=bound_arguments or {},
        exc=exc,
        duration_ms=duration_ms,
    )


def _bind_arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _capture_success_event(
    telemetry: Telemetry,
    *,
    spec: OperationTelemetrySpec,
    bound_arguments: dict[str, Any],
    result: Any,
) -> None:
    if spec.success_builder is None or spec.success_event is None:
        return
    with suppress(Exception):
        properties = {
            "invocation_surface": invocation_surface().value,
            **spec.success_builder(bound_arguments, result),
        }
        telemetry.capture(spec.success_event, properties)


def _capture_operation_result(
    telemetry: Telemetry,
    *,
    spec: OperationTelemetrySpec,
    bound_arguments: dict[str, Any],
    exc: BaseException | None,
    duration_ms: int,
) -> None:
    with suppress(Exception):
        identifier = identifier_type(
            bound_arguments.get(spec.identifier_param)
            if spec.identifier_param
            else None
        )
        failure_reason = failure_reason_from_exception(exc)
        properties = _drop_none(
            {
                "invocation_surface": invocation_surface().value,
                "operation_name": spec.operation_name.value,
                "operation_group": spec.operation_group.value,
                "target_resource": (
                    spec.target_resource.value
                    if spec.target_resource is not None
                    else None
                ),
                "identifier_type": (
                    identifier.value if identifier is not None else None
                ),
                "result": command_result_from_exception(exc).value,
                "failure_reason": (
                    failure_reason.value
                    if failure_reason is not None
                    else None
                ),
                "duration_ms": duration_ms,
            }
        )
        telemetry.capture(
            OPERATION_RESULT_EVENT,
            properties,
            include_system_metadata=True,
        )


def _wrap_cyclopts(app: cyclopts.App, *, prefix: str) -> None:
    default_command = app.default_command
    if default_command is not None and not _is_builtin_cyclopts_command(
        app, default_command
    ):
        command_path = prefix or _primary_name(app.name)
        metadata = command_metadata(command_path)
        if metadata is not None:
            app.default_command = _wrap_command_callback(
                default_command,
                command_name=metadata.command_name,
                command_group=metadata.command_group,
            )

    for subapp in _iter_unique_subapps(app._commands.values()):
        name = _primary_name(subapp.name)
        if not name or name.startswith("-"):
            continue
        subapp_prefix = name if not prefix else f"{prefix} {name}"
        _wrap_cyclopts(subapp, prefix=subapp_prefix)


def _wrap_command_callback(
    func: Callable[..., Any],
    *,
    command_name: CommandName,
    command_group: TelemetryGroup,
) -> Callable[..., Any]:
    if getattr(func, "_hubai_telemetry_wrapped", False):
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            caught_exc: BaseException | None = None
            try:
                return await func(*args, **kwargs)
            except BaseException as exc:
                caught_exc = exc
                raise
            finally:
                _emit_command_event(
                    command_name=command_name,
                    command_group=command_group,
                    exc=caught_exc,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        cast(Any, async_wrapper)._hubai_telemetry_wrapped = True
        return async_wrapper

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        caught_exc: BaseException | None = None
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            caught_exc = exc
            raise
        finally:
            _emit_command_event(
                command_name=command_name,
                command_group=command_group,
                exc=caught_exc,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    cast(Any, wrapper)._hubai_telemetry_wrapped = True
    return wrapper


def _emit_command_event(
    *,
    command_name: CommandName,
    command_group: TelemetryGroup,
    exc: BaseException | None,
    duration_ms: int,
) -> None:
    if is_suppressed():
        return
    telemetry = get_component_telemetry()
    failure_reason = pop_cli_failure_reason() or failure_reason_from_exception(
        exc
    )
    telemetry.capture(
        COMMAND_EVENT,
        build_command_properties(
            command_name=command_name,
            command_group=command_group,
            result=command_result_from_exception(exc),
            failure_reason=failure_reason,
            duration_ms=duration_ms,
        ),
        include_system_metadata=True,
    )


def command_metadata(command_path: str) -> CommandMetadata | None:
    normalized = " ".join(command_path.split())
    if normalized == "login":
        return CommandMetadata(CommandName.LOGIN, TelemetryGroup.AUTH)
    if normalized == "logout":
        return CommandMetadata(CommandName.LOGOUT, TelemetryGroup.AUTH)
    if normalized == "convert":
        return CommandMetadata(CommandName.CONVERT, TelemetryGroup.CONVERSION)
    mapping = {
        "model ls": CommandMetadata(
            CommandName.MODEL_LIST, TelemetryGroup.MODELS
        ),
        "model info": CommandMetadata(
            CommandName.MODEL_INFO, TelemetryGroup.MODELS
        ),
        "model create": CommandMetadata(
            CommandName.MODEL_CREATE, TelemetryGroup.MODELS
        ),
        "model update": CommandMetadata(
            CommandName.MODEL_UPDATE, TelemetryGroup.MODELS
        ),
        "model delete": CommandMetadata(
            CommandName.MODEL_DELETE, TelemetryGroup.MODELS
        ),
        "variant ls": CommandMetadata(
            CommandName.VARIANT_LIST, TelemetryGroup.VARIANTS
        ),
        "variant info": CommandMetadata(
            CommandName.VARIANT_INFO, TelemetryGroup.VARIANTS
        ),
        "variant create": CommandMetadata(
            CommandName.VARIANT_CREATE, TelemetryGroup.VARIANTS
        ),
        "variant delete": CommandMetadata(
            CommandName.VARIANT_DELETE, TelemetryGroup.VARIANTS
        ),
        "instance ls": CommandMetadata(
            CommandName.INSTANCE_LIST, TelemetryGroup.INSTANCES
        ),
        "instance info": CommandMetadata(
            CommandName.INSTANCE_INFO, TelemetryGroup.INSTANCES
        ),
        "instance download": CommandMetadata(
            CommandName.INSTANCE_DOWNLOAD, TelemetryGroup.INSTANCES
        ),
        "instance create": CommandMetadata(
            CommandName.INSTANCE_CREATE, TelemetryGroup.INSTANCES
        ),
        "instance delete": CommandMetadata(
            CommandName.INSTANCE_DELETE, TelemetryGroup.INSTANCES
        ),
        "instance config": CommandMetadata(
            CommandName.INSTANCE_CONFIG, TelemetryGroup.INSTANCES
        ),
        "instance files": CommandMetadata(
            CommandName.INSTANCE_FILES, TelemetryGroup.INSTANCES
        ),
        "instance upload": CommandMetadata(
            CommandName.INSTANCE_UPLOAD, TelemetryGroup.INSTANCES
        ),
    }
    return mapping.get(normalized)


def _iter_unique_subapps(subapps: Iterable[object]) -> list[cyclopts.App]:
    unique: list[cyclopts.App] = []
    seen: set[int] = set()
    for subapp in subapps:
        if not isinstance(subapp, cyclopts.App):
            continue
        identity = id(subapp)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(subapp)
    return unique


def _primary_name(name: tuple[str, ...] | str | None) -> str:
    if name is None:
        return ""
    if isinstance(name, tuple):
        return name[0] if name else ""
    return name


def _is_builtin_cyclopts_command(
    app: cyclopts.App, default_command: object
) -> bool:
    return default_command in {app.help_print, app.version_print}


def _failure_reason_from_status_code(
    status_code: int | None,
) -> FailureReason:
    if status_code in {401, 403}:
        return FailureReason.AUTH_ERROR
    if status_code == 404:
        return FailureReason.NOT_FOUND
    if status_code == 409:
        return FailureReason.CONFLICT
    if status_code in {400, 422}:
        return FailureReason.VALIDATION_ERROR
    if status_code in {408, 504}:
        return FailureReason.TIMEOUT
    if status_code is not None and status_code >= 500:
        return FailureReason.SERVER_ERROR
    if status_code is not None:
        return FailureReason.API_ERROR
    return FailureReason.UNKNOWN


def _drop_none(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _sdk_version() -> str | None:
    try:
        return version("hubai-sdk")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def build_models_listed_properties(
    arguments: dict[str, Any], result: list[Any]
) -> dict[str, Any]:
    tasks = arguments.get("tasks") or []
    return {
        "has_task_filter": bool(tasks),
        "task_filter_count_bucket": (
            bucket_nonzero_count(len(tasks)) if tasks else None
        ),
        "license_type_filter": arguments.get("license_type"),
        "visibility_filter": visibility_filter(
            arguments.get("is_public")
        ).value,
        "has_project_filter": arguments.get("project_id") is not None,
        "luxonis_only": bool(arguments.get("luxonis_only")),
        "limit_bucket": bucket_limit(int(arguments["limit"])),
        "sort_mode": sort_mode(
            str(arguments["sort"]),
            allowed={"updated", "name", "id", "created"},
        ),
        "sort_order": arguments.get("order"),
        "result_count_bucket": bucket_result_count(len(result)),
    }


def build_model_identifier_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    identifier = identifier_type(arguments.get("identifier"))
    return {
        "identifier_type": identifier.value
        if identifier is not None
        else None,
    }


def build_model_created_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    tasks = arguments.get("tasks") or []
    links = arguments.get("links") or []
    description_short = arguments.get("description_short")
    return {
        "license_type": arguments.get("license_type"),
        "visibility": visibility_value(arguments.get("is_public")).value,
        "has_description": bool(arguments.get("description")),
        "has_description_short": bool(
            description_short and description_short != "<empty>"
        ),
        "has_architecture_id": arguments.get("architecture_id") is not None,
        "task_count_bucket": bucket_zero_count(len(tasks)),
        "link_count_bucket": bucket_zero_count(len(links)),
        "is_yolo": bool(arguments.get("is_yolo")),
    }


def build_model_updated_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    updates = {
        "license_type": arguments.get("license_type") is not None,
        "visibility": arguments.get("is_public") is not None,
        "description": arguments.get("description") is not None,
        "description_short": arguments.get("description_short") is not None,
        "architecture_id": arguments.get("architecture_id") is not None,
        "tasks": arguments.get("tasks") is not None,
        "links": arguments.get("links") is not None,
        "is_yolo": arguments.get("is_yolo") is not None,
    }
    updated_count = sum(updates.values())
    tasks = arguments.get("tasks") or []
    links = arguments.get("links") or []
    description_short = arguments.get("description_short")
    identifier = identifier_type(arguments.get("identifier"))
    return _drop_none(
        {
            "identifier_type": identifier.value
            if identifier is not None
            else None,
            "visibility": (
                visibility_value(arguments.get("is_public")).value
                if updates["visibility"]
                else None
            ),
            "license_type": arguments.get("license_type"),
            "has_description": (
                bool(arguments.get("description"))
                if updates["description"]
                else None
            ),
            "has_description_short": (
                bool(description_short and description_short != "<empty>")
                if updates["description_short"]
                else None
            ),
            "has_architecture_id": (
                arguments.get("architecture_id") is not None
                if updates["architecture_id"]
                else None
            ),
            "task_count_bucket": (
                bucket_zero_count(len(tasks)) if updates["tasks"] else None
            ),
            "link_count_bucket": (
                bucket_zero_count(len(links)) if updates["links"] else None
            ),
            "is_yolo": arguments.get("is_yolo"),
            "updated_field_count_bucket": bucket_zero_count(updated_count),
            "updated_license_type": updates["license_type"],
            "updated_visibility": updates["visibility"],
            "updated_description": updates["description"],
            "updated_description_short": updates["description_short"],
            "updated_architecture_id": updates["architecture_id"],
            "updated_tasks": updates["tasks"],
            "updated_links": updates["links"],
            "updated_is_yolo": updates["is_yolo"],
        }
    )


def build_variants_listed_properties(
    arguments: dict[str, Any], result: list[Any]
) -> dict[str, Any]:
    model_identifier = identifier_type(arguments.get("model_id"))
    return {
        "has_model_filter": arguments.get("model_id") is not None,
        "model_identifier_type": (
            model_identifier.value if model_identifier is not None else None
        ),
        "has_name_filter": arguments.get("name") is not None,
        "has_variant_slug_filter": arguments.get("variant_slug") is not None,
        "has_version_filter": arguments.get("variant_version") is not None,
        "visibility_filter": visibility_filter(
            arguments.get("is_public")
        ).value,
        "include_model_name": bool(arguments.get("include_model_name")),
        "limit_bucket": bucket_limit(int(arguments["limit"])),
        "sort_mode": sort_mode(
            str(arguments["sort"]),
            allowed={"updated", "name", "id", "created", "version"},
        ),
        "sort_order": arguments.get("order"),
        "result_count_bucket": bucket_result_count(len(result)),
    }


def build_variant_created_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    tags = arguments.get("tags") or []
    model_identifier = identifier_type(arguments.get("model_id"))
    return {
        "model_identifier_type": (
            model_identifier.value if model_identifier is not None else None
        ),
        "has_description": bool(arguments.get("description")),
        "has_repository_url": bool(arguments.get("repository_url")),
        "has_commit_hash": bool(arguments.get("commit_hash")),
        "has_domain": bool(arguments.get("domain")),
        "tag_count_bucket": bucket_zero_count(len(tags)),
    }


def build_instances_listed_properties(
    arguments: dict[str, Any], result: list[Any]
) -> dict[str, Any]:
    platforms = arguments.get("platforms") or []
    model_identifier = identifier_type(arguments.get("model_id"))
    variant_identifier = identifier_type(arguments.get("variant_id"))
    parent_identifier = identifier_type(arguments.get("parent_id"))
    model_type = arguments.get("model_type")
    return _drop_none(
        {
            "platform_filter_count_bucket": (
                bucket_nonzero_count(len(platforms)) if platforms else None
            ),
            "has_search_filter": arguments.get("search") is not None,
            "has_model_filter": arguments.get("model_id") is not None,
            "model_identifier_type": (
                model_identifier.value
                if model_identifier is not None
                else None
            ),
            "has_variant_filter": arguments.get("variant_id") is not None,
            "variant_identifier_type": (
                variant_identifier.value
                if variant_identifier is not None
                else None
            ),
            "model_type": (
                model_type.value.lower() if model_type is not None else None
            ),
            "has_parent_filter": arguments.get("parent_id") is not None,
            "parent_identifier_type": (
                parent_identifier.value
                if parent_identifier is not None
                else None
            ),
            "model_class": arguments.get("model_class"),
            "has_name_filter": arguments.get("name") is not None,
            "has_hash_filter": arguments.get("hash") is not None,
            "status": arguments.get("status"),
            "compression_level": arguments.get("compression_level"),
            "optimization_level": arguments.get("optimization_level"),
            "visibility_filter": visibility_filter(
                arguments.get("is_public")
            ).value,
            "include_model_name": bool(arguments.get("include_model_name")),
            "limit_bucket": bucket_limit(int(arguments["limit"])),
            "sort_mode": sort_mode(
                str(arguments["sort"]),
                allowed={"updated", "name", "id", "created", "status"},
            ),
            "sort_order": arguments.get("order"),
            "result_count_bucket": bucket_result_count(len(result)),
        }
    )


def build_instance_created_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    tags = arguments.get("tags") or []
    quantization_data = arguments.get("quantization_data")
    model_type = arguments.get("model_type")
    variant_identifier = identifier_type(arguments.get("variant_id"))
    parent_identifier = identifier_type(arguments.get("parent_id"))
    return _drop_none(
        {
            "variant_identifier_type": (
                variant_identifier.value
                if variant_identifier is not None
                else None
            ),
            "has_parent_instance": arguments.get("parent_id") is not None,
            "parent_identifier_type": (
                parent_identifier.value
                if parent_identifier is not None
                else None
            ),
            "model_type": (
                model_type.value.lower() if model_type is not None else None
            ),
            "quantization_mode": arguments.get("quantization_mode"),
            "quantization_input_type": quantization_input_type(
                quantization_data
            ).value,
            "tag_count_bucket": bucket_zero_count(len(tags)),
            "input_shape_provided": arguments.get("input_shape") is not None,
            "is_deployable": arguments.get("is_deployable"),
            "yolo_version": arguments.get("yolo_version"),
        }
    )


def build_instance_files_listed_properties(
    arguments: dict[str, Any], result: list[Any]
) -> dict[str, Any]:
    identifier = identifier_type(arguments.get("identifier"))
    return {
        "identifier_type": identifier.value
        if identifier is not None
        else None,
        "result_count_bucket": bucket_zero_count(len(result)),
    }


def build_instance_downloaded_properties(
    arguments: dict[str, Any], result: Path
) -> dict[str, Any]:
    downloaded_count = 1 if result.exists() else 0
    identifier = identifier_type(arguments.get("identifier"))
    return {
        "identifier_type": identifier.value
        if identifier is not None
        else None,
        "output_dir_provided": arguments.get("output_dir") is not None,
        "force": bool(arguments.get("force")),
        "downloaded_file_count_bucket": bucket_nonzero_count(downloaded_count),
    }


def build_instance_uploaded_properties(
    arguments: dict[str, Any], _result: Any
) -> dict[str, Any]:
    path = Path(arguments["file_path"])
    identifier = identifier_type(arguments.get("identifier"))
    return {
        "identifier_type": identifier.value
        if identifier is not None
        else None,
        "file_extension": file_extension(path),
        "file_size_bucket": bucket_file_size(path.stat().st_size),
    }
