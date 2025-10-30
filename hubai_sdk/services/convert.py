from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.typing import Kwargs, PathType

from hubai_sdk.services.instances import (
    create_instance,
    download_instance,
    upload_file,
)
from hubai_sdk.services.models import create_model
from hubai_sdk.services.variants import create_variant
from hubai_sdk.typing import (
    License,
    Quantization,
    TargetPrecision,
    Task,
    YoloVersion,
)
from hubai_sdk.utils.constants import SHARED_DIR
from hubai_sdk.utils.hub import (
    get_configs,
    get_resource_id,
    get_target_specific_options,
    get_variant_name,
    get_version_number,
    wait_for_export,
)
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.hubai_models import ModelInstanceResponse
from hubai_sdk.utils.nn_archive import cleanup_extracted_path
from hubai_sdk.utils.sdk_models import ConvertResponse
from hubai_sdk.utils.types import InputFileType, ModelType, PotDevice, Target
from hubai_sdk.services.variants import get_variant


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
    target_precision: TargetPrecision = "INT8",
    domain: str | None = None,
    variant_tags: list[str] | None = None,
    variant_id: UUID | str | None = None,
    quantization_data: Quantization | None = None,
    max_quantization_images: int | None = None,
    instance_tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    output_dir: str | None = None,
    tool_version: str | None = None,
    yolo_input_shape: str | None = None,
    yolo_version: YoloVersion | None = None,
    yolo_class_names: list[str] | None = None,
) -> ConvertResponse:
    """Starts the online conversion process.

    Parameters
    ----------
    target : Target
        The target platform.
    path : str
        Path to the model file, NN Archive, or configuration file.
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : UUID | str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : UUID | str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    variant_version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    max_quantization_images : int, optional
        Maximum number of quantization images.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : UUID | str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion. For RVC2 & RVC3, this is the IR version while for RVC4, this is the SNPE version.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
    opts : list[str], optional
        Additional options for the conversion process.
    """

    logger.info(f"Converting model to {target.name} format")
    logger.info(f"Options: {opts}")

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
        input_file_type = InputFileType.from_path(path)
        if input_file_type == InputFileType.PYTORCH and yolo_version is None:
            raise ValueError(
                "YOLO version is required for PyTorch YOLO models. Use --yolo-version to specify the version."
            )

    if target_precision in {"FP16", "FP32"}:
        opts.extend(["disable_calibration", "True"])

    if yolo_input_shape:
        opts.extend(["yolo_input_shape", str(yolo_input_shape)])

    config_path = None
    if path and (is_archive or is_yaml(path)):
        config_path = path

    cfg, *_ = get_configs(config_path, opts)
    cleanup_extracted_path(SHARED_DIR)

    if len(cfg.stages) > 1:
        raise ValueError(
            "Only single-stage models are supported with online conversion."
        )

    name = name or cfg.name

    cfg = next(iter(cfg.stages.values()))

    model_type = ModelType.from_suffix(cfg.input_model.suffix)
    variant_name = get_variant_name(cfg, model_type, name)

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
                silent=True,
            )
            assert model is not None
            model_id = model.id
        except ValueError:
            model_id = get_resource_id(
                name.lower().replace(" ", "-"), "models"
            )

    if variant_id is None:
        if model_id is None:
            raise ValueError("`--model-id` is required to create a new model")

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
            silent=True,
        )
        assert variant is not None
        variant_id = variant.id

    else:
        variant = get_variant(variant_id)
        if variant is None:
            raise ValueError(f"Variant with ID {variant_id} not found")
        if variant_version is not None:
            if model_id is None:
                raise ValueError("`--model-id` is required to create a new variant version.")
            variant = create_variant(
                variant.name,
                model_id=model_id,
                variant_version=variant_version,
                description=variant_description,
                repository_url=repository_url,
                commit_hash=commit_hash,
                domain=domain,
                tags=variant_tags or [],
                silent=True,
            )
            assert variant is not None
            variant_id = variant.id

    assert variant_id is not None
    instance_name = f"{variant_name} base instance"
    instance = create_instance(
        instance_name,
        variant_id=variant_id,
        model_type=model_type,
        input_shape=input_shape or cfg.inputs[0].shape,
        is_deployable=is_deployable,
        tags=instance_tags or [],
        silent=True,
    )
    assert instance is not None
    instance_id = instance.id

    # TODO: IR support
    if path is not None and is_nn_archive(path):
        logger.info(f"Uploading NN archive: {path}")
        upload_file(path, instance_id)
    else:
        logger.info(f"Uploading input model: {cfg.input_model}")
        upload_file(str(cfg.input_model), instance_id)

    if target is Target.RVC4 and quantization_data is None:
        quantization_data = (
            None if target_precision in {"FP16", "FP32"} else "RANDOM"
        )

    target_options = get_target_specific_options(target, cfg, tool_version)
    instance = _export(
        f"{variant_name} exported to {target}",
        instance_id,
        target=target,
        target_precision=target_precision or "INT8",
        quantization_data=quantization_data.upper()
        if quantization_data
        else None,
        max_quantization_images=max_quantization_images,
        yolo_version=yolo_version,
        yolo_class_names=yolo_class_names,
        **target_options,
    )

    wait_for_export(str(instance.dag_run_id))

    downloaded_path = download_instance(instance.id, output_dir)

    return ConvertResponse(downloaded_path=downloaded_path, instance=instance)


def _export(
    name: str,
    identifier: UUID | str,
    target: Target,
    target_precision: str,
    quantization_data: str | None,
    max_quantization_images: int | None = None,
    yolo_version: str | None = None,
    yolo_class_names: list[str] | None = None,
    **kwargs,
) -> ModelInstanceResponse:
    """Exports a model instance."""
    model_instance_id = get_resource_id(str(identifier), "modelInstances")
    json: dict[str, Any] = {
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
        json["target_precision"] = target_precision
    res = Request.post(
        service="models",
        endpoint=f"modelInstances/{model_instance_id}/export/{target.value}",
        json=json,
        params={
            "legacy": not json.get("superblob", True) and target is Target.RVC2
        },
    )
    logger.info(
        f"Model instance '{name}' created for {target.name} export with ID '{res['id']}'"
    )
    return ModelInstanceResponse(**res)


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

    Parameters
    ----------
    path : PathType
        Path to the model file to be converted.
    mo_args : list[str] | None, optional
        Additional arguments for the Model Optimizer (MO).
    compile_tool_args : list[str] | None, optional
        Additional arguments for the compile tool.
    compress_to_fp16 : bool, default True
        Whether to compress the model's weights to FP16.
    number_of_shaves : int, default 8
        Number of shaves to use for the conversion.
    superblob : bool, default True
        Whether to create a superblob for the model.
    opts : dict[str, Any] | list[str] | None, optional
        Additional options for the conversion. Can be used
        to override configuration values.
    **hub_kwargs
        Additional keyword arguments to be passed to the
        online conversion. See also `convert` function.

    Hub kwargs:
    ------------
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : UUID | str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : UUID | str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    variant_version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    max_quantization_images : int, optional
        Maximum number of quantization images.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : UUID | str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion. For RVC2 & RVC3, this is the IR version while for RVC4, this is the SNPE version.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
    """
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

    Parameters
    ----------
    path : PathType
        Path to the model file to be converted.
    mo_args : list[str] | None, optional
        Additional arguments for the Model Optimizer (MO).
    compile_tool_args : list[str] | None, optional
        Additional arguments for the compile tool.
    compress_to_fp16 : bool, default True
        Whether to compress the model's weights to FP16.
    pot_target_device : PotDevice | Literal["VPU", "ANY"], default PotDevice.VPU
        Target device for POT quantization.
    opts : dict[str, Any] | list[str] | None, optional
        Additional options for the conversion. Can be used
        to override configuration values.
    **hub_kwargs
        Additional keyword arguments to be passed to the
        online conversion.

    Hub kwargs:
    ------------
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : UUID | str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : UUID | str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    variant_version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    max_quantization_images : int, optional
        Maximum number of quantization images.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : UUID | str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion. For RVC2 & RVC3, this is the IR version while for RVC4, this is the SNPE version.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
    """
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

    Parameters
    ----------
    path : PathType
        Path to the model file to be converted.
    snpe_onnx_to_dlc_args : list[str] | None, optional
        Additional arguments for the SNPE ONNX to DLC conversion.
    snpe_dlc_quant_args : list[str] | None, optional
        Additional arguments for the SNPE DLC quantization.
    snpe_dlc_graph_prepare_args : list[str] | None, optional
        Additional arguments for the SNPE DLC graph preparation.
    use_per_channel_quantization : bool, default True
        Whether to use per-channel quantization.
    use_per_row_quantization : bool, default False
        Whether to use per-row quantization.
    htp_socs : list[str] | None, optional
        List of HTP SoCs for the final DLC graph.
    opts : dict[str, Any] | list[str] | None, optional
        Additional options for the conversion. Can be used
        to override configuration values.
    **hub_kwargs
        Additional keyword arguments to be passed to the
        online conversion. See also `convert` function.

    Hub kwargs:
    ------------
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : UUID | str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : UUID | str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    variant_version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    max_quantization_images : int, optional
        Maximum number of quantization images.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : UUID | str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion. For RVC2 & RVC3, this is the IR version while for RVC4, this is the SNPE version.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
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

    Parameters
    ----------
    path : PathType
        Path to the model file to be converted.
    optimization_level : int, default 2
        Optimization level for the conversion.
    compression_level : int, default 2
        Compression level for the conversion.
    batch_size : int, default 8
        Batch size for the conversion.
    alls : list[str] | None, optional
        List of `alls` parameters for the conversion.
    opts : dict[str, Any] | list[str] | None, optional
        Additional options for the conversion. Can be used
        to override configuration values.
    **hub_kwargs
        Additional keyword arguments to be passed to the
        online conversion. See also `convert` function.

    Hub kwargs:
    ------------
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : UUID | str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : UUID | str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    variant_version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    max_quantization_images : int, optional
        Maximum number of quantization images.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : UUID | str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion. For RVC2 & RVC3, this is the IR version while for RVC4, this is the SNPE version.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
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
