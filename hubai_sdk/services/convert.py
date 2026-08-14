"""Run hosted model conversion and download deployable artifacts.

The target-specific helpers `RVC2`, `RVC3`, `RVC4`, and `Hailo` are the
recommended conversion API. Each helper prepares target options and delegates
to `convert`, which coordinates the complete HubAI workflow:

    1. Read model metadata from a model file, YAML config, or NN Archive.
    2. Create or reuse model and variant resources.
    3. Upload the source model as an instance.
    4. Start an export job and wait for it to finish.
    5. Download the exported instance.

Example:
    Convert an ONNX model for RVC4 with HubAI's general calibration data.

    .. python::

        result = client.convert.RVC4(
            path="detector.onnx",
            name="detector",
            quantization_mode="INT8_STANDARD",
            quantization_data="GENERAL",
        )
        print(result.downloaded_path)

Input configuration:
    ``path`` may point to a model file, an NN Archive, or a YAML configuration.
    Additional ``opts`` override values loaded from that configuration. A
    dictionary is usually easiest in Python; the flat list form exists for the
    command-line interface.

Resource reuse:
    Pass ``model_id`` or ``variant_id`` to attach the conversion to existing
    resources. Without them, `convert` creates resources and reuses a model
    with the same slug if one already exists.

Quantization:
    RVC4 and Hailo accept a built-in calibration domain, a dataset ID beginning
    with ``aid_``, or a local ZIP file. Passing a ZIP path uploads it for the
    export job. RVC2 and RVC3 ignore quantization inputs because those helpers
    expose their platform-specific precision controls instead.

Conversion parameters
---------------------

The conversion API accepts parameters for the input, HubAI resources, and the
selected target. The function signatures below provide exact types and defaults;
this overview groups the parameters by purpose.

General parameters
^^^^^^^^^^^^^^^^^^

* ``path``: Input model file, YAML configuration, or NN Archive.
* ``tool_version``: Version of the conversion tool.
* ``quantization_mode``: RVC4 quantization mode. ``FP16_STANDARD`` skips
  calibration.
* ``output_dir``: Directory for downloaded output files. If omitted, the
  downloader creates a directory named after the exported instance slug in the
  current working directory.

YOLO parameters
^^^^^^^^^^^^^^^

* ``yolo_input_shape``: YOLO input shape. It defaults to ``[640, 640]`` when
  omitted or invalid.
* ``yolo_version``: YOLO version, such as ``"yolov8"``. Prefer auto-detection
  and set this only when detection fails.
* ``yolo_class_names``: Model class names.

Model parameters
^^^^^^^^^^^^^^^^

These parameters create a HubAI Model resource unless ``model_id`` is supplied.

* ``model_id``: ID of an existing model to reuse.
* ``name``: Model name. It defaults to the configured or input model name.
* ``license_type``: License type for the model.
* ``is_public``: Whether the model is public, private, or team-scoped.
* ``description`` and ``description_short``: Full and short model
  descriptions. The short description defaults to ``"<empty>"``.
* ``architecture_id``: Model architecture ID.
* ``tasks``: Tasks the model supports.
* ``links``: Additional links for the model.
* ``is_yolo``: Whether the model is a YOLO model.

Model variant parameters
^^^^^^^^^^^^^^^^^^^^^^^^

* ``model_id``: ID of an existing model to which to add a variant.
* ``variant_version``: Version of the model variant. HubAI uses the next
  version automatically when it is omitted.
* ``variant_description``: Full variant description.
* ``repository_url`` and ``commit_hash``: Related repository URL and commit.
* ``domain``: Variant domain.
* ``variant_tags``: Variant tags.

Model instance parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``variant_id``: ID of an existing model variant to reuse.
* ``quantization_data``: Predefined calibration domain, an ``aid_`` dataset ID,
  or a local custom ``.zip`` path. Pass the ZIP path rather than ``CUSTOM``.
* ``max_quantization_images``: Maximum number of images for quantization.
* ``instance_tags``: Model-instance tags.
* ``input_shape``: Model input shape.
* ``is_deployable``: Whether the model instance is deployable.

RVC2 parameters
^^^^^^^^^^^^^^^

* ``mo_args``: Arguments for the Model Optimizer.
* ``compile_tool_args``: Arguments for the BLOB compiler.
* ``compress_to_fp16``: Whether to compress model weights to FP16. Defaults to
  ``True``.
* ``number_of_shaves``: Number of shaves to use. Defaults to ``8``.
* ``superblob``: Whether to create a superblob. Defaults to ``True``; disable
  it to use legacy RVC2 conversion.

RVC3 parameters
^^^^^^^^^^^^^^^

* ``mo_args``: Arguments for the Model Optimizer.
* ``compile_tool_args``: Arguments for the BLOB compiler.
* ``compress_to_fp16``: Whether to compress model weights to FP16. Defaults to
  ``True``.
* ``pot_target_device``: Target device for post-training optimization. Defaults
  to ``"VPU"``.

RVC4 parameters
^^^^^^^^^^^^^^^

* ``snpe_onnx_to_dlc_args``: Arguments for ``snpe-onnx-to-dlc``.
* ``snpe_dlc_quant_args``: Arguments for ``snpe-dlc-quant``.
* ``snpe_dlc_graph_prepare_args``: Arguments for ``snpe-dlc-graph-prepare``.
* ``use_per_channel_quantization``: Whether to use per-channel quantization.
  Defaults to ``True``.
* ``use_per_row_quantization``: Whether to use per-row quantization. Defaults
  to ``False``.
* ``htp_socs``: HTP SoCs to use. Defaults to ``["sm8550"]``.

Hailo parameters
^^^^^^^^^^^^^^^^

* ``optimization_level``: Optimization level. Defaults to ``2``.
* ``compression_level``: Compression level. Defaults to ``2``.
* ``batch_size``: Quantization batch size. Defaults to ``8``.
* ``alls``: Additional ALLS parameters.
"""

import time
from pathlib import Path
from time import sleep
from typing import Literal
from uuid import UUID

from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.telemetry import suppress_telemetry
from luxonis_ml.typing import Kwargs, PathType
from requests import HTTPError

from hubai_sdk.errors import (
    HubApiError,
    InputError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from hubai_sdk.services.instances import (
    create_instance,
    download_instance,
    upload_file,
    upload_quantization_zip,
)
from hubai_sdk.services.models import create_model
from hubai_sdk.services.variants import create_variant, get_variant
from hubai_sdk.typing import (
    License,
    QuantizationData,
    QuantizationMode,
    Task,
    YoloVersion,
)
from hubai_sdk.utils.constants import SHARED_DIR
from hubai_sdk.utils.hub import (
    get_configs,
    get_resource_info,
    get_target_specific_options,
    get_variant_name,
    get_version_number,
    raise_for_hub_error,
    resolve_resource_id,
    wait_for_job,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import (
    EnumModelInstanceStatus,
)
from hubai_sdk.utils.nn_archive import cleanup_extracted_path
from hubai_sdk.utils.quantization import normalize_quantization_input
from hubai_sdk.utils.sdk_models import (
    ConvertResponse,
    JobMessageResponse,
    ModelInstanceResponse,
)
from hubai_sdk.utils.telemetry import (
    ConversionPhase,
    FailureReason,
    OperationName,
    OperationTelemetrySpec,
    TargetResource,
    TelemetryGroup,
    TelemetryResult,
    build_conversion_result_properties,
    build_conversion_summary,
    capture_conversion_configured,
    capture_conversion_result,
    config_source_from_path,
    conversion_failure_reason,
    current_conversion_run_id,
    get_or_create_conversion_run_id,
    invocation_surface,
    quantization_input_type,
    reset_conversion_run_id,
    telemetry_operation,
)
from hubai_sdk.utils.types import ModelType, PotDevice, Target


@telemetry_operation(
    OperationTelemetrySpec(
        operation_name=OperationName.CONVERT,
        operation_group=TelemetryGroup.CONVERSION,
        target_resource=TargetResource.CONVERSION,
    )
)
def convert(
    target: Target,
    opts: list[str] | None = None,
    /,
    *,
    path: str,
    name: str | None = None,
    license_type: License = "undefined",
    is_public: bool | None = False,
    description_short: str = "<empty>",
    description: str | None = None,
    architecture_id: UUID | str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool = False,
    model_id: UUID | str | None = None,
    variant_version: str | None = None,
    variant_description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    quantization_mode: QuantizationMode | None = None,
    domain: str | None = None,
    variant_tags: list[str] | None = None,
    variant_id: UUID | str | None = None,
    quantization_data: QuantizationData | PathType | None = None,
    max_quantization_images: int | None = None,
    instance_tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    output_dir: str | None = None,
    tool_version: str | None = None,
    yolo_input_shape: list[int] | None = None,
    yolo_version: YoloVersion | None = None,
    yolo_class_names: list[str] | None = None,
) -> ConvertResponse:
    """Starts the online conversion process.

    Args:
        target: Target platform.
        path: Path to the input model file, NN Archive, or YAML configuration
            file.
        name: Model name. If not specified, the name is taken from the
            configuration file or the model file.
        license_type: License type for a newly created model.
        is_public: Whether the model is public (``True``), private
            (``False``), or team-scoped (``None``).
        description_short: Short description of the model. Defaults to
            ``"<empty>"``.
        description: Full description of the model.
        architecture_id: Architecture ID.
        tasks: Tasks the model supports.
        links: Additional links for the model.
        is_yolo: Whether the model is a YOLO model.
        model_id: ID of an existing model resource. If specified, that
            model is used instead of creating a new one.
        variant_version: Model version. If not specified, the version is
            auto-incremented from the latest version of the model. If no
            versions exist, the version is ``"0.1.0"``.
        variant_description: Full description of the model variant.
        repository_url: URL of the repository.
        commit_hash: Commit hash.
        quantization_mode: Quantization mode to use during RVC4 conversion.
            Must be one of ``INT8_STANDARD``,
            ``INT8_ACCURACY_FOCUSED``, ``INT8_INT16_MIXED``,
            ``INT8_INT16_MIXED_ACCURACY_FOCUSED``, or
            ``FP16_STANDARD``. ``INT8_STANDARD`` is standard INT8
            quantization with calibration for optimal performance and
            model size. ``INT8_ACCURACY_FOCUSED`` is INT8 quantization
            with calibration that may improve accuracy without reducing
            performance or increasing model size, depending on the
            model. ``INT8_INT16_MIXED`` uses 8-bit weights and 16-bit
            activations across all layers for improved numeric
            stability and accuracy at the cost of performance and model
            size. ``INT8_INT16_MIXED_ACCURACY_FOCUSED`` is a mixed INT8
            and INT16 calibration-based mode that prioritizes accuracy
            over throughput. ``FP16_STANDARD`` is FP16 quantization
            without calibration for models that require higher accuracy
            and numeric stability at the cost of performance and model
            size. ``FP16_STANDARD`` skips calibration.
        domain: Domain of the model variant.
        variant_tags: Tags for the model variant.
        variant_id: ID of an existing model version resource. If
            specified, that version is used instead of creating a new
            one.
        quantization_data: Data used to quantize this model. This can be
            a predefined domain (``DRIVING``, ``FOOD``, ``GENERAL``,
            ``INDOORS``, ``RANDOM``, ``WAREHOUSE``, ``CLIP``, ``UNKNOWN``), a
            dataset ID, or a path to a local quantization ``.zip`` file.
            Pass the ``.zip`` path itself instead of ``CUSTOM``; the SDK
            normalizes local zip inputs automatically.
        max_quantization_images: Maximum number of images to use for
            quantization.
        instance_tags: Tags for the model instance.
        input_shape: Input shape for the model instance.
        is_deployable: Whether the model instance is deployable.
        output_dir: Directory path for the downloaded files. If not
            specified, the downloader creates a directory named after
            the exported instance slug under the current working
            directory.
        tool_version: Version of the tool used for conversion. For
            RVC2 and RVC3 this is the IR version, while for RVC4 this
            is the SNPE version.
        yolo_input_shape: Input shape for YOLO models. It defaults to
            ``[640, 640]`` when omitted or invalid.
        yolo_version: YOLO version, such as ``"yolov8"``. Usually omit this
            to use auto-detection; specify it only if detection fails.
        yolo_class_names: Class names for YOLO models.
        opts: Additional options for the conversion process.

    Returns:
        Conversion result containing the downloaded output path, export
        job, and exported model instance.
    """
    logger.info(f"Converting model to {target.name} format")
    logger.info(f"Options: {opts}")

    previous_conversion_run_id = current_conversion_run_id()
    conversion_run_id = get_or_create_conversion_run_id()
    start = time.monotonic()
    configured_properties: dict[str, object] | None = None
    emitted_configured_event = False
    phase = ConversionPhase.CONFIGURATION
    response: ConvertResponse | None = None

    if isinstance(architecture_id, UUID):
        architecture_id = str(architecture_id)
    if isinstance(model_id, UUID):
        model_id = str(model_id)
    if isinstance(variant_id, UUID):
        variant_id = str(variant_id)

    opts = opts or []

    is_archive = is_nn_archive(path)

    def is_yaml(path: str) -> bool:
        return Path(path).suffix in [".yaml", ".yml"]

    if path is not None and not is_archive and not is_yaml(path):
        opts.extend(["input_model", path])

    if quantization_mode in {"FP16_STANDARD", "FP32_STANDARD"}:
        opts.extend(["disable_calibration", "True"])

    if yolo_input_shape:
        opts.extend(["yolo_input_shape", str(yolo_input_shape)])

    config_path = None
    if path and (is_archive or is_yaml(path)):
        config_path = path

    cfg, *_ = get_configs(config_path, opts)
    cleanup_extracted_path(SHARED_DIR)

    if len(cfg.stages) > 1:
        raise InputError(
            "Only single-stage models are supported with online conversion."
        )

    name = name or cfg.name

    cfg = next(iter(cfg.stages.values()))

    model_type = ModelType.from_suffix(cfg.input_model.suffix)
    variant_name = get_variant_name(cfg, model_type, name)

    existing_model_reused = model_id is not None
    existing_variant_reused = (
        variant_id is not None and variant_version is None
    )

    normalized_quantization_input = normalize_quantization_input(
        quantization_data
    )
    quantization_data = normalized_quantization_input.quantization_data
    custom_quantization_zip = normalized_quantization_input.custom_zip_path

    if target is Target.RVC4 and quantization_data is None:
        quantization_data = (
            None
            if quantization_mode in {"FP16_STANDARD", "FP32_STANDARD"}
            else "GENERAL"
        )

    try:
        with suppress_telemetry():
            if model_id is None and variant_id is None:
                try:
                    model = create_model(
                        name,
                        license_type=license_type,
                        is_public=is_public,
                        description=description,
                        description_short=description_short,
                        architecture_id=architecture_id,
                        tasks=tasks or [],
                        links=links or [],
                        is_yolo=is_yolo,
                    )
                    model_id = model.id
                except ResourceConflictError as conflict_error:
                    model_identifier = name.lower().replace(" ", "-")
                    try:
                        model_id = resolve_resource_id(
                            model_identifier, "models"
                        )
                    except ResourceNotFoundError as lookup_error:
                        raise ResourceConflictError(
                            f"Unable to create model '{name}': HubAI "
                            f"reported a resource conflict ({conflict_error}). "
                            "The SDK then tried to reuse an existing model "
                            f"using identifier '{model_identifier}', but it "
                            f"could not be resolved ({lookup_error}).",
                            status_code=conflict_error.status_code,
                        ) from lookup_error
                    existing_model_reused = True

            if variant_id is None:
                if model_id is None:
                    raise InputError(
                        "`--model-id` is required to create a new model"
                    )

                version = variant_version or get_version_number(str(model_id))

                variant = create_variant(
                    variant_name,
                    model_id=model_id,
                    variant_version=version,
                    description=variant_description,
                    repository_url=repository_url,
                    commit_hash=commit_hash,
                    domain=domain,
                    tags=variant_tags or [],
                )
                variant_id = variant.id

            else:
                variant = get_variant(variant_id)
                if variant_version is not None:
                    if model_id is None:
                        raise InputError(
                            "`--model-id` is required to create a new variant version."
                        )
                    variant = create_variant(
                        variant.name,
                        model_id=model_id,
                        variant_version=variant_version,
                        description=variant_description,
                        repository_url=repository_url,
                        commit_hash=commit_hash,
                        domain=domain,
                        tags=variant_tags or [],
                    )
                    variant_id = variant.id
                else:
                    existing_variant_reused = True

        target_options = get_target_specific_options(target, cfg, tool_version)
        configured_properties = build_conversion_summary(
            target=target.value,
            config_source=config_source_from_path(
                path,
                is_archive=is_archive,
                is_yaml=bool(
                    config_path
                    and Path(config_path).suffix in {".yaml", ".yml"}
                ),
            ),
            input_model_type=model_type,
            invocation_surface=invocation_surface(),
            existing_model_reused=existing_model_reused,
            existing_variant_reused=existing_variant_reused,
            quantization_mode=quantization_mode,
            quantization_input_type=quantization_input_type(
                quantization_data,
                custom_zip=custom_quantization_zip is not None,
            ),
            max_quantization_images=max_quantization_images,
            yolo_version=yolo_version,
            yolo_class_names=yolo_class_names,
            yolo_input_shape_provided=yolo_input_shape is not None,
            tool_version_provided=tool_version is not None,
            input_shape_provided=input_shape is not None,
            download_output_dir_provided=output_dir is not None,
            input_count=len(cfg.inputs),
            target_options=target_options,
        )
        capture_conversion_configured(
            conversion_run_id,
            configured_properties,
        )
        emitted_configured_event = True

        with suppress_telemetry():
            assert variant_id is not None
            instance_name = f"{variant_name} base instance"
            phase = ConversionPhase.RESOURCE_SETUP
            instance = create_instance(
                instance_name,
                variant_id=variant_id,
                model_type=model_type,
                input_shape=input_shape or cfg.inputs[0].shape,
                is_deployable=is_deployable,
                tags=instance_tags or [],
            )
            instance_id = instance.id

            phase = ConversionPhase.UPLOAD
            if path is not None and is_nn_archive(path):
                logger.info(f"Uploading NN archive: {path}")
                upload_file(path, instance_id)
            else:
                logger.info(f"Uploading input model: {cfg.input_model}")
                upload_file(str(cfg.input_model), instance_id)

            export_name = f"{variant_name} exported to {target}"
            phase = ConversionPhase.EXPORT
            export_job = _export(
                export_name,
                instance_id,
                target=target,
                quantization_mode=quantization_mode or "INT8_STANDARD",
                quantization_data=quantization_data,
                max_quantization_images=max_quantization_images,
                yolo_version=yolo_version,
                yolo_class_names=yolo_class_names,
                **target_options,
            )

            if custom_quantization_zip is not None:
                phase = ConversionPhase.UPLOAD
                logger.info(
                    f"Uploading custom quantization zip: {custom_quantization_zip}"
                )
                upload_quantization_zip(
                    str(custom_quantization_zip), export_job.id
                )
                phase = ConversionPhase.EXPORT

            export_job = wait_for_job(export_job.id)
            instance = _resolve_exported_instance(export_job)

            phase = ConversionPhase.DOWNLOAD
            downloaded_path = download_instance(instance.id, output_dir)
        response = ConvertResponse(
            downloaded_path=downloaded_path,
            job=export_job,
            instance=instance,
        )
    except BaseException as exc:
        if emitted_configured_event:
            capture_conversion_result(
                conversion_run_id,
                {
                    **(configured_properties or {"target": target.value}),
                    **build_conversion_result_properties(
                        result=(
                            TelemetryResult.INTERRUPTED
                            if conversion_failure_reason(exc, phase=phase)
                            == FailureReason.USER_INTERRUPT
                            else TelemetryResult.FAILED
                        ),
                        failure_reason=conversion_failure_reason(
                            exc, phase=phase
                        ),
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                },
            )
        raise
    else:
        if emitted_configured_event:
            capture_conversion_result(
                conversion_run_id,
                {
                    **(configured_properties or {"target": target.value}),
                    **build_conversion_result_properties(
                        result=TelemetryResult.SUCCESS,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                },
            )
        assert response is not None
        return response
    finally:
        reset_conversion_run_id(previous_conversion_run_id)


def _resolve_exported_instance(
    job: JobMessageResponse,
) -> ModelInstanceResponse:
    """Resolve the exported model instance from a completed export job.

    Returns:
        The exported model instance once it is ready to download.
    """
    instance_id = job.result["resulting_model_instance_id"]
    if not isinstance(instance_id, str):
        raise TypeError("resulting_model_instance_id must be a string")
    return _wait_for_exported_instance_ready(instance_id)


def _get_instance_response(instance_id: str) -> ModelInstanceResponse | None:
    """Fetch a model instance response, returning ``None`` if missing.

    Returns:
        The model instance response, or ``None`` if the instance does not
        exist yet.
    """
    try:
        data = get_resource_info(instance_id, "modelInstances")
    except ResourceNotFoundError:
        return None
    return ModelInstanceResponse(**data)


def _wait_for_exported_instance_ready(
    instance_id: str,
    *,
    timeout_seconds: int = 180,
    poll_interval_seconds: int = 2,
) -> ModelInstanceResponse:
    """Wait until an exported model instance is actually downloadable.

    Returns:
        The exported model instance once it becomes downloadable.
    """
    attempts = max(1, timeout_seconds // poll_interval_seconds)
    latest_instance = None
    last_error: Exception | None = None

    for _ in range(attempts):
        latest_instance = _get_instance_response(instance_id)
        if latest_instance is None:
            sleep(poll_interval_seconds)
            continue

        is_available = (
            latest_instance.status == EnumModelInstanceStatus.available
        )

        if is_available:
            try:
                if Request.get(
                    service="models",
                    endpoint=f"modelInstances/{latest_instance.id}/download",
                ):
                    return latest_instance
            except HTTPError as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )
                if status_code not in {404, 409, 423}:
                    raise_for_hub_error(
                        exc,
                        identifier=latest_instance.id,
                        endpoint="modelInstances",
                    )
                last_error = exc

        sleep(poll_interval_seconds)

    if last_error is not None:
        raise HubApiError(
            "Export job completed but the exported model instance is not "
            "downloadable yet."
        ) from last_error

    raise HubApiError(
        "Export job completed but the exported model instance did not "
        "become available in time."
    )


def _export(
    name: str,
    identifier: UUID | str,
    target: Target,
    quantization_mode: QuantizationMode | None,
    quantization_data: str | None,
    max_quantization_images: int | None = None,
    yolo_version: str | None = None,
    yolo_class_names: list[str] | None = None,
    **kwargs,
) -> JobMessageResponse:
    """Starts an export job for a model instance.

    Returns:
        The created export job response.
    """
    model_instance_id = resolve_resource_id(str(identifier), "modelInstances")
    json: dict[str, object] = {
        "name": name,
        "quantization_data": quantization_data,
        "max_quantization_images": max_quantization_images,
        **kwargs,
    }
    if yolo_version:
        json["version"] = yolo_version
    if yolo_class_names:
        json["class_names"] = yolo_class_names
    if yolo_version and not yolo_class_names:
        logger.warning(
            "It's recommended to provide YOLO class names via --yolo-class-names. If omitted, class names will be extracted from model weights if present, otherwise default names will be used."
        )
    if target is Target.RVC4:
        json["target_precision"] = quantization_mode
    try:
        res = Request.post(
            service="models",
            endpoint=f"modelInstances/{model_instance_id}/export/{target.value}",
            json=json,
            params={
                "legacy": not json.get("superblob", True)
                and target is Target.RVC2
            },
        )
    except HTTPError as exc:
        raise_for_hub_error(
            exc,
            identifier=model_instance_id,
            endpoint="modelInstances",
        )
    job = JobMessageResponse(**res)
    logger.info(
        f"Export job '{job.id}' created for model instance '{model_instance_id}' "
        f"targeting {target.name}"
    )
    return job


def RVC2(
    path: PathType,
    mo_args: list[str] | None = None,
    compile_tool_args: list[str] | None = None,
    compress_to_fp16: bool = True,
    number_of_shaves: int = 8,
    superblob: bool = True,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> ConvertResponse:
    """Convert a model to RVC2 format.

    Args:
        path: Path to the model file to convert.
        mo_args: Additional arguments for the Model Optimizer.
        compile_tool_args: Additional arguments for the compile tool.
        compress_to_fp16: Whether to compress model weights to FP16.
            Defaults to ``True``.
        number_of_shaves: Number of shaves to use for the conversion. Defaults
            to ``8``.
        superblob: Whether to create a superblob for the model. Defaults to
            ``True``. Disable it to use legacy RVC2 conversion.
        opts: Additional conversion options. These can override config
            values.
        **hub_kwargs: Additional keyword arguments passed to `convert`.

    Keyword Args:
        name: Model name. If not specified, the name is taken from the
            configuration file or the model file.
        license_type: License type.
        is_public: Whether the model is public (``True``), private
            (``False``), or team-scoped (``None``).
        description_short: Short description of the model.
        description: Full description of the model.
        architecture_id: Architecture ID.
        tasks: Tasks this model supports.
        links: Links to related resources.
        is_yolo: Whether the model is a YOLO model.
        model_id: ID of an existing model resource. If specified, that
            model is used instead of creating a new one.
        variant_version: Model version. If not specified, the version is
            auto-incremented from the latest version of the model. If no
            versions exist, the version is ``"0.1.0"``.
        variant_description: Full description of the model variant.
        repository_url: URL of the repository.
        commit_hash: Commit hash.
        domain: Domain of the model.
        variant_tags: Tags for the model variant.
        variant_id: ID of an existing model version resource. If
            specified, that version is used instead of creating a new
            one.
        input_shape: Input shape for the model instance.
        is_deployable: Whether the model instance is deployable.
        output_dir: Directory path for the downloaded files. If not
            specified, the downloader creates a directory named after
            the exported instance slug under the current working
            directory.
        tool_version: Version of the tool used for conversion. For
            RVC2 and RVC3 this is the IR version, while for RVC4 this
            is the SNPE version.
        yolo_input_shape: Input shape for YOLO models.
        yolo_version: YOLO version.
        yolo_class_names: Class names for YOLO models.

    Returns:
        Conversion result containing the downloaded output path, export
        job, and exported model instance.
    """
    if hub_kwargs.get("quantization_mode") is not None:
        logger.warning(
            "`quantization_mode` is not supported for RVC2. It will be ignored."
        )
        del hub_kwargs["quantization_mode"]

    if hub_kwargs.get("quantization_data") is not None:
        logger.warning(
            "`quantization_data` is not supported for RVC2. It will be ignored."
        )
        del hub_kwargs["quantization_data"]

    if hub_kwargs.get("max_quantization_images") is not None:
        logger.warning(
            "`max_quantization_images` is not supported for RVC2. It will be ignored."
        )
        del hub_kwargs["max_quantization_images"]

    return convert(
        Target.RVC2,
        _combine_opts(
            Target.RVC2,
            {
                "mo_args": mo_args or [],
                "compile_tool_args": compile_tool_args or [],
                "compress_to_fp16": compress_to_fp16,
                "number_of_shaves": number_of_shaves,
                "superblob": superblob,
            },
            opts,
        ),
        path=str(path),
        **hub_kwargs,
    )


def RVC3(
    path: PathType,
    mo_args: list[str] | None = None,
    compile_tool_args: list[str] | None = None,
    compress_to_fp16: bool = True,
    pot_target_device: PotDevice | Literal["VPU", "ANY"] = PotDevice.VPU,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> ConvertResponse:
    """Convert a model to RVC3 format.

    Args:
        path: Path to the model file to convert.
        mo_args: Additional arguments for the Model Optimizer.
        compile_tool_args: Additional arguments for the compile tool.
        compress_to_fp16: Whether to compress model weights to FP16.
            Defaults to ``True``.
        pot_target_device: Target device for post-training optimization.
            Defaults to ``"VPU"``.
        opts: Additional conversion options. These can override config
            values.
        **hub_kwargs: Additional keyword arguments passed to `convert`.

    Keyword Args:
        name: Model name. If not specified, the name is taken from the
            configuration file or the model file.
        license_type: License type.
        is_public: Whether the model is public (``True``), private
            (``False``), or team-scoped (``None``).
        description_short: Short description of the model.
        description: Full description of the model.
        architecture_id: Architecture ID.
        tasks: Tasks this model supports.
        links: Links to related resources.
        is_yolo: Whether the model is a YOLO model.
        model_id: ID of an existing model resource. If specified, that
            model is used instead of creating a new one.
        variant_version: Model version. If not specified, the version is
            auto-incremented from the latest version of the model. If no
            versions exist, the version is ``"0.1.0"``.
        variant_description: Full description of the model variant.
        repository_url: URL of the repository.
        commit_hash: Commit hash.
        domain: Domain of the model.
        variant_tags: Tags for the model variant.
        variant_id: ID of an existing model version resource. If
            specified, that version is used instead of creating a new
            one.
        input_shape: Input shape for the model instance.
        is_deployable: Whether the model instance is deployable.
        output_dir: Directory path for the downloaded files. If not
            specified, the downloader creates a directory named after
            the exported instance slug under the current working
            directory.
        tool_version: Version of the tool used for conversion. For
            RVC2 and RVC3 this is the IR version, while for RVC4 this
            is the SNPE version.
        yolo_input_shape: Input shape for YOLO models.
        yolo_version: YOLO version.
        yolo_class_names: Class names for YOLO models.

    Returns:
        Conversion result containing the downloaded output path, export
        job, and exported model instance.
    """
    if hub_kwargs.get("quantization_mode") is not None:
        logger.warning(
            "`quantization_mode` is not supported for RVC3. It will be ignored."
        )
        del hub_kwargs["quantization_mode"]

    if hub_kwargs.get("quantization_data") is not None:
        logger.warning(
            "`quantization_data` is not supported for RVC3. It will be ignored."
        )
        del hub_kwargs["quantization_data"]

    if hub_kwargs.get("max_quantization_images") is not None:
        logger.warning(
            "`max_quantization_images` is not supported for RVC3. It will be ignored."
        )
        del hub_kwargs["max_quantization_images"]

    if not isinstance(pot_target_device, PotDevice):
        pot_target_device = PotDevice(pot_target_device)
    return convert(
        Target.RVC3,
        _combine_opts(
            Target.RVC3,
            {
                "mo_args": mo_args or [],
                "compile_tool_args": compile_tool_args or [],
                "compress_to_fp16": compress_to_fp16,
                "pot_target_device": pot_target_device.value,
            },
            opts,
        ),
        path=str(path),
        **hub_kwargs,
    )


def RVC4(
    path: PathType,
    snpe_onnx_to_dlc_args: list[str] | None = None,
    snpe_dlc_quant_args: list[str] | None = None,
    snpe_dlc_graph_prepare_args: list[str] | None = None,
    use_per_channel_quantization: bool = True,
    use_per_row_quantization: bool = False,
    htp_socs: list[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ]
    | None = None,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> ConvertResponse:
    """Convert a model to RVC4 format.

    Args:
        path: Path to the model file to convert.
        snpe_onnx_to_dlc_args: Additional arguments for the SNPE ONNX to
            DLC conversion.
        snpe_dlc_quant_args: Additional arguments for SNPE DLC
            quantization.
        snpe_dlc_graph_prepare_args: Additional arguments for SNPE DLC
            graph preparation.
        use_per_channel_quantization: Whether to use per-channel
            quantization. Defaults to ``True``.
        use_per_row_quantization: Whether to use per-row quantization.
            Defaults to ``False``.
        htp_socs: HTP SoCs for the final DLC graph. Defaults to
            ``["sm8550"]``.
        opts: Additional conversion options. These can override config
            values.
        **hub_kwargs: Additional keyword arguments passed to `convert`.

    Keyword Args:
        name: Model name. If not specified, the name is taken from the
            configuration file or the model file.
        license_type: License type.
        is_public: Whether the model is public (``True``), private
            (``False``), or team-scoped (``None``).
        description_short: Short description of the model.
        description: Full description of the model.
        architecture_id: Architecture ID.
        tasks: Tasks this model supports.
        links: Links to related resources.
        is_yolo: Whether the model is a YOLO model.
        model_id: ID of an existing model resource. If specified, that
            model is used instead of creating a new one.
        variant_version: Model version. If not specified, the version is
            auto-incremented from the latest version of the model. If no
            versions exist, the version is ``"0.1.0"``.
        variant_description: Full description of the model variant.
        repository_url: URL of the repository.
        commit_hash: Commit hash.
        quantization_mode: Quantization mode to use during conversion.
            Must be one of ``INT8_STANDARD``,
            ``INT8_ACCURACY_FOCUSED``, ``INT8_INT16_MIXED``,
            ``INT8_INT16_MIXED_ACCURACY_FOCUSED``, or
            ``FP16_STANDARD``. ``INT8_STANDARD`` is standard INT8
            quantization with calibration for optimal performance and
            model size. ``INT8_ACCURACY_FOCUSED`` is INT8 quantization
            with calibration that may improve accuracy without reducing
            performance or increasing model size, depending on the
            model. ``INT8_INT16_MIXED`` uses 8-bit weights and 16-bit
            activations across all layers for improved numeric
            stability and accuracy at the cost of performance and model
            size. ``INT8_INT16_MIXED_ACCURACY_FOCUSED`` is a mixed INT8
            and INT16 calibration-based mode that prioritizes accuracy
            over throughput. ``FP16_STANDARD`` is FP16 quantization
            without calibration for models that require higher accuracy
            and numeric stability at the cost of performance and model
            size.
        domain: Domain of the model.
        variant_tags: Tags for the model variant.
        variant_id: ID of an existing model version resource. If
            specified, that version is used instead of creating a new
            one.
        quantization_data: Data used to quantize this model. This can be
            a predefined domain (``DRIVING``, ``FOOD``, ``GENERAL``,
            ``INDOORS``, ``RANDOM``, ``WAREHOUSE``, ``CLIP``, ``UNKNOWN``), a
            dataset ID, or a path to a local quantization ``.zip`` file.
            Pass the ``.zip`` path itself instead of ``CUSTOM``; the SDK
            normalizes local zip inputs automatically.
        max_quantization_images: Maximum number of quantization images.
        input_shape: Input shape for the model instance.
        is_deployable: Whether the model instance is deployable.
        output_dir: Directory path for the downloaded files. If not
            specified, the downloader creates a directory named after
            the exported instance slug under the current working
            directory.
        tool_version: Version of the tool used for conversion. For
            RVC2 and RVC3 this is the IR version, while for RVC4 this
            is the SNPE version.
        yolo_input_shape: Input shape for YOLO models.
        yolo_version: YOLO version.
        yolo_class_names: Class names for YOLO models.

    Returns:
        Conversion result containing the downloaded output path, export
        job, and exported model instance.
    """
    htp_socs = htp_socs or ["sm8550"]
    return convert(
        Target.RVC4,
        _combine_opts(
            Target.RVC4,
            {
                "snpe_onnx_to_dlc_args": snpe_onnx_to_dlc_args or [],
                "snpe_dlc_quant_args": snpe_dlc_quant_args or [],
                "snpe_dlc_graph_prepare_args": snpe_dlc_graph_prepare_args
                or [],
                "use_per_channel_quantization": use_per_channel_quantization,
                "use_per_row_quantization": use_per_row_quantization,
                "htp_socs": htp_socs,
            },
            opts,
        ),
        path=str(path),
        **hub_kwargs,
    )


def Hailo(
    path: PathType,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2,
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2,
    batch_size: int = 8,
    alls: list[str] | None = None,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> ConvertResponse:
    """Convert a model to Hailo format.

    Args:
        path: Path to the model file to convert.
        optimization_level: Optimization level for the conversion. Defaults to
            ``2``.
        compression_level: Compression level for the conversion. Defaults to
            ``2``.
        batch_size: Batch size for quantization. Defaults to ``8``.
        alls: Additional Hailo ALLS parameters to apply during conversion.
        opts: Additional conversion options. These can override config
            values.
        **hub_kwargs: Additional keyword arguments passed to `convert`.

    Keyword Args:
        name: Model name. If not specified, the name is taken from the
            configuration file or the model file.
        license_type: License type.
        is_public: Whether the model is public (``True``), private
            (``False``), or team-scoped (``None``).
        description_short: Short description of the model.
        description: Full description of the model.
        architecture_id: Architecture ID.
        tasks: Tasks this model supports.
        links: Links to related resources.
        is_yolo: Whether the model is a YOLO model.
        model_id: ID of an existing model resource. If specified, that
            model is used instead of creating a new one.
        variant_version: Model version. If not specified, the version is
            auto-incremented from the latest version of the model. If no
            versions exist, the version is ``"0.1.0"``.
        variant_description: Full description of the model variant.
        repository_url: URL of the repository.
        commit_hash: Commit hash.
        quantization_mode: Quantization mode.
        quantization_data: Data used to quantize this model. This can be
            a predefined domain (``DRIVING``, ``FOOD``, ``GENERAL``,
            ``INDOORS``, ``RANDOM``, ``WAREHOUSE``, ``CLIP``, ``UNKNOWN``), a
            dataset ID, or a path to a local quantization ``.zip`` file.
        max_quantization_images: Maximum number of quantization images.
        domain: Domain of the model.
        variant_tags: Tags for the model variant.
        variant_id: ID of an existing model version resource. If
            specified, that version is used instead of creating a new
            one.
        input_shape: Input shape for the model instance.
        is_deployable: Whether the model instance is deployable.
        output_dir: Directory path for the downloaded files. If not
            specified, the downloader creates a directory named after
            the exported instance slug under the current working
            directory.
        tool_version: Version of the tool used for conversion. For
            RVC2 and RVC3 this is the IR version, while for RVC4 this
            is the SNPE version.
        yolo_input_shape: Input shape for YOLO models.
        yolo_version: YOLO version.
        yolo_class_names: Class names for YOLO models.

    Returns:
        Conversion result containing the downloaded output path, export
        job, and exported model instance.
    """
    return convert(
        Target.HAILO,
        _combine_opts(
            Target.HAILO,
            {
                "optimization_level": optimization_level,
                "compression_level": compression_level,
                "batch_size": batch_size,
                "alls": alls or [],
            },
            opts,
        ),
        path=str(path),
        **hub_kwargs,
    )


def _combine_opts(
    target: Target, target_kwargs: Kwargs, opts: list[str] | Kwargs | None
) -> list[str]:
    """Merge generic options with target-prefixed conversion options.

    Returns:
        A flat option list ready for config parsing.
    """
    opts = opts or []
    if isinstance(opts, dict):
        opts_list = []
        for key, value in opts.items():
            opts_list.extend([key, value])
    else:
        opts_list = opts

    for key, value in target_kwargs.items():
        opts_list.extend([f"{target.value}.{key}", value])

    return opts_list
