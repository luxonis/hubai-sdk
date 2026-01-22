# Conversion Parameters

The Python API for conversion is available through `HubAIClient.convert` namespace. In this document we will list and describe all the parameters that can be used for conversion. Besides conversion-specific parameters, the methods also accept additional parameters about the HubAI model, variant etc.

## Table of Contents

- [General parameters](#general-parameters)
- [YOLO parameters](#yolo-parameters)
- [Model parameters](#model-parameters)
- [Model variant parameters](#model-variant-parameters)
- [Model instance parameters](#model-instance-parameters)
- [RVC2 parameters](#rvc2-parameters)
- [RVC3 parameters](#rvc3-parameters)
- [RVC4 parameters](#rvc4-parameters)
- [Hailo parameters](#hailo-parameters)

## General parameters

General parameters applicable to all conversion functions.

| argument            | type                                                                                                      | description                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `path`              | `str`                                                                                                     | The path to the model file.                                                                                            |
| `tool_version`      | `str \| None`                                                                                             | The version of the conversion tool.                                                                                    |
| `quantization_mode` | `Literal["INT8_STANDARD", "INT8_ACCURACY_FOCUSED", "INT8_INT16_MIXED", "FP16_STANDARD", "FP32_STANDARD"]` | The quantization mode of the model. Defaults to `"INT8_STANDARD"`. Only applicable for RVC4                            |
| `output_dir`        | `str \| None`                                                                                             | The directory to save the converted model. If not specified, the model will be saved in the current working directory. |

## YOLO Parameters

These parameters are only relevant if you're converting a YOLO model.

| argument           | type                | description                        |
| ------------------ | ------------------- | ---------------------------------- |
| `yolo_input_shape` | `list[int] \| None` | The input shape of the YOLO model. |
| `yolo_version`     | `str \| None`       | YOLO version (e.g. "yolov8").      |
| `yolo_class_names` | `list[str] \| None` | The class names of the model.      |

## Model Parameters

Parameters that specify creation of a new `Model` resource on HubAI.

| argument            | type                                                                                                                                                                                                                                                                               | description                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `model_id`          | `str \| None`                                                                                                                                                                                                                                                                      | The ID of an already existing model in case you don't want to create a new `Model` resource. |
| `name`              | `str \| None`                                                                                                                                                                                                                                                                      | The name of the model. If undefined, it will be the same as the stem of the model file.      |
| `license_type`      | `Literal["undefined", "MIT", "GNU General Public License v3.0", "GNU Affero General Public License v3.0", "Apache 2.0", "NTU S-Lab 1.0", "Ultralytics Enterprise", "CreativeML Open RAIL-M", "BSD 3-Clause"]`                                                                      | The license type of the model.                                                               |
| `is_public`         | `bool \| None`                                                                                                                                                                                                                                                                     | Whether the model is public or private.                                                      |
| `description`       | `str \| None`                                                                                                                                                                                                                                                                      | The full description of the model.                                                           |
| `description_short` | `str \| None`                                                                                                                                                                                                                                                                      | The short description of the model. Defaults to `"<empty>"`                                  |
| `architecture_id`   | `str \| None`                                                                                                                                                                                                                                                                      | The architecture ID of the model.                                                            |
| `tasks`             | `list[Literal["CLASSIFICATION", "OBJECT_DETECTION", "SEGMENTATION", "KEYPOINT_DETECTION", "DEPTH_ESTIMATION", "LINE_DETECTION", "FEATURE_DETECTION", "DENOISING", "LOW_LIGHT_ENHANCEMENT", "SUPER_RESOLUTION", "REGRESSION", "INSTANCE_SEGMENTATION", "IMAGE_EMBEDDING"]] \| None` | The tasks of the model.                                                                      |
| `links`             | `list[str] \| None`                                                                                                                                                                                                                                                                | Additional links for the model.                                                              |
| `is_yolo`           | `bool \| None`                                                                                                                                                                                                                                                                     | Whether the model is a YOLO model.                                                           |

## Model Variant Parameters

Parameters that specify creation of a new `ModelVersion` resource on HubAI.

| argument              | type                | description                                                                                    |
| --------------------- | ------------------- | ---------------------------------------------------------------------------------------------- |
| `model_id`            | `str \| None`       | The ID of the model. Use in case you want to add another variant to an already existing model. |
| `variant_version`     | `str \| None`       | The version number of the variant. If undefined, an auto-incremented version is used.          |
| `variant_description` | `str \| None`       | The full description of the variant.                                                           |
| `repository_url`      | `str \| None`       | A URL of a related repository.                                                                 |
| `commit_hash`         | `str \| None`       | A commit hash of the related repository.                                                       |
| `domain`              | `str \| None`       | The domain of the variant.                                                                     |
| `variant_tags`        | `list[str] \| None` | The tags of the variant.                                                                       |

## Model Instance Parameters

Parameters that specify creation of a new `ModelInstance` resource on HubAI.

| argument                  | type                                                                              | description                                                                                                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `variant_id`              | `str`                                                                             | The ID of the associated variant. Use in case you want to add a new model instance to an already existing model variant or if you want to add a new version of the variant. |
| `parent_id`               | `str \| None`                                                                     | Unique identifier of the parent `ModelInstance`.                                                                                                                            |
| `quantization_data`       | `Literal["DRIVING", "FOOD", "GENERAL", "INDOORS", "RANDOM", "WAREHOUSE"] \| None` | The domain of data used to quantize this `ModelInstance`.                                                                                                                   |
| `max_quantization_images` | `int \| None`                                                                     | The maximum number of images to use for quantization.                                                                                                                       |
| `instance_tags`           | `list[str] \| None`                                                               | Tags associated with this instance.                                                                                                                                         |
| `input_shape`             | `list[int] \| None`                                                               | The input shape of the model.                                                                                                                                               |
| `is_deployable`           | `bool \| None`                                                                    | Whether the model is deployable.                                                                                                                                            |

## RVC2 Parameters

Parameters specific to the `RVC2` conversion.

| argument            | type                | description                                                                                       |
| ------------------- | ------------------- | ------------------------------------------------------------------------------------------------- |
| `mo_args`           | `list[str] \| None` | The arguments to pass to the model optimizer.                                                     |
| `compile_tool_args` | `list[str] \| None` | The arguments to pass to the BLOB compiler.                                                       |
| `compress_to_fp16`  | `bool`              | Whether to compress the model's weights to FP16 precision. Defaults to `True`.                    |
| `number_of_shaves`  | `int`               | The number of shaves to use. Defaults to `8`.                                                     |
| `superblob`         | `bool`              | Whether to create a superblob. Defaults to `True`. Disable it if you want legacy RVC2 conversion. |

## RVC3 Parameters

Parameters specific to the `RVC3` conversion.

| argument            | type                    | description                                                                    |
| ------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| `mo_args`           | `list[str] \| None`     | The arguments to pass to the model optimizer.                                  |
| `compile_tool_args` | `list[str] \| None`     | The arguments to pass to the BLOB compiler.                                    |
| `compress_to_fp16`  | `bool`                  | Whether to compress the model's weights to FP16 precision. Defaults to `True`. |
| `pot_target_device` | `Literal["VPU", "ANY"]` | The target device for the post-training optimization. Defaults to `"VPU"`.     |

## RVC4 Parameters

Parameters specific to the `RVC4` conversion.

| argument                       | type                | description                                                  |
| ------------------------------ | ------------------- | ------------------------------------------------------------ |
| `snpe_onnx_to_dlc_args`        | `list[str] \| None` | The arguments to pass to the `snpe-onnx-to-dlc` tool.        |
| `snpe_dlc_quant_args`          | `list[str] \| None` | The arguments to pass to the `snpe-dlc-quant` tool.          |
| `snpe_dlc_graph_prepare_args`  | `list[str] \| None` | The arguments to pass to the `snpe-dlc-graph-prepare` tool.  |
| `use_per_channel_quantization` | `bool`              | Whether to use per-channel quantization. Defaults to `True`. |
| `use_per_row_quantization`     | `bool`              | Whether to use per-row quantization. Defaults to `False`.    |
| `htp_socs`                     | `list[str] \| None` | The list of HTP SoCs to use.                                 |

## Hailo Parameters

Parameters specific to the `Hailo` conversion.

| argument             | type                           | description                                      |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| `optimization_level` | `Literal[-100, 0, 1, 2, 3, 4]` | The optimization level to use.                   |
| `compression_level`  | `Literal[0, 1, 2, 3, 4, 5]`    | The compression level to use.                    |
| `batch_size`         | `int`                          | The batch size to use for quantization.          |
| `alls`               | `list[str] \| None`            | The list of additional `alls` parameters to use. |
