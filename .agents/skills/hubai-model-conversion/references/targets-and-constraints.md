# Targets and Constraints

## Supported Inputs

- `ONNX`: `.onnx`
- `OpenVINO IR`: `.xml` and optional `.bin`
- `TFLite`: `.tflite`
- `PyTorch YOLO`: `.pt`, `.pth`
- `NN Archive`: `.tar.xz`
- `Config-driven conversion`: `.yaml`, `.yml`

PyTorch inputs require `yolo_version`.

## Common Hub Metadata

Use these only when the task needs Hub-side resource management in addition to conversion:

- `name`
- `license_type`
- `is_public`
- `description_short`
- `description`
- `architecture_id`
- `tasks`
- `links`
- `model_id`
- `variant_id`
- `variant_version`
- `variant_description`
- `repository_url`
- `commit_hash`
- `domain`
- `variant_tags`
- `instance_tags`
- `input_shape`
- `is_deployable`
- `output_dir`

If no existing model or variant IDs are supplied, the SDK creates the Hub resources it needs and auto-increments the variant version.

## RVC2

Preferred helper: `client.convert.RVC2(...)`

Primary knobs:

- `number_of_shaves`
- `superblob`
- `mo_args`
- `compile_tool_args`

Behavior notes:

- `superblob=False` is the legacy blob path.
- `quantization_mode`, `quantization_data`, and `max_quantization_images` are ignored.

## RVC3

Preferred helper: `client.convert.RVC3(...)`

Primary knobs:

- `compress_to_fp16`
- `pot_target_device`
- `mo_args`
- `compile_tool_args`

Behavior notes:

- `quantization_mode`, `quantization_data`, and `max_quantization_images` are ignored.

## RVC4

Preferred helper: `client.convert.RVC4(...)`

Primary knobs:

- `quantization_mode`
- `quantization_data`
- `max_quantization_images`
- `snpe_onnx_to_dlc_args`
- `snpe_dlc_quant_args`
- `snpe_dlc_graph_prepare_args`
- `use_per_channel_quantization`
- `use_per_row_quantization`
- `htp_socs`

Supported quantization modes:

- `INT8_STANDARD`
- `INT8_ACCURACY_FOCUSED`
- `INT8_INT16_MIXED`
- `INT8_INT16_MIXED_ACCURACY_FOCUSED`
- `FP16_STANDARD`
- `FP32_STANDARD`

Accepted `quantization_data` forms:

- predefined dataset name such as `GENERAL`, `INDOORS`, or `WAREHOUSE`
- dataset ID beginning with `aid_`
- path to a custom `.zip` file

Behavior notes:

- When `quantization_data` is omitted, the SDK defaults to `RANDOM` unless the mode is `FP16_STANDARD` or `FP32_STANDARD`.
- Passing `CUSTOM` directly is an error. Pass the `.zip` path instead.

## Hailo

Preferred helper: `client.convert.Hailo(...)`

Primary knobs:

- `optimization_level`
- `compression_level`
- `batch_size`
- `alls`

## Global Constraints

- Hosted conversion supports only single-stage models.
- The SDK uploads the source artifact to Hub and downloads the exported instance when the job completes.
- The returned object includes `downloaded_path`, the export `job`, and the exported `instance`.

## Blobconverter Migration Hints

Common mappings when replacing `blobconverter`:

- `model` -> `path`
- `xml` -> `path`
- `bin` -> `opts["input_bin"]`
- `version` -> `tool_version`
- `data_type` -> `quantization_mode`
- `shaves` -> `number_of_shaves`
- `optimizer_params` -> `mo_args`
- `compile_params` -> `compile_tool_args`
