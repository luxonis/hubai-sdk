# Targets and Constraints

## Supported Inputs

- `ONNX`: `.onnx`
- `OpenVINO IR`: `.xml` and optional `.bin`
- `TFLite`: `.tflite`
- `PyTorch YOLO`: `.pt`, `.pth`
- `NN Archive`: `.tar.xz`
- `Config-driven conversion`: `.yaml`, `.yml`

All of these go through the same `path` argument.

PyTorch inputs are limited to YOLO models. Normally rely on auto-detected `yolo_version`. `yolo_input_shape` defaults to `[640, 640]` when omitted or invalid.

## Common Conversion and Hub Parameters

Use these only when the task needs more than a simple one-off conversion:

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

`output_dir` is a directory path. If it is omitted, the downloader creates a directory named from the exported instance slug under the current working directory.

If the conversion created a fresh model hierarchy and the caller only wants local files afterward, prefer cleanup by deleting the created model via `client.models.delete_model(response.instance.model_id)` or `hubai model delete <model_id>`.

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

These settings are RVC4-only.

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

Accepted `quantization_data` forms:

- predefined dataset name such as `GENERAL`, `INDOORS`, or `WAREHOUSE`
- dataset ID beginning with `aid_`
- path to a local existing custom `.zip` file

Behavior notes:

- On calibration-based modes, omitted `quantization_data` defaults to `RANDOM`.
- `FP16_STANDARD` does not use calibration data.
- Passing `CUSTOM` directly is an error. Pass the `.zip` path instead; the SDK will normalize that input to `CUSTOM`.

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
- OpenVINO IR can use `path` for the XML and `input_bin` for the BIN when they do not live together.

## Blobconverter Migration Hints

Common mappings when replacing `blobconverter`:

- `model` -> `path`
- `xml` -> `path`
- `bin` -> `opts["input_bin"]`
- `version` -> `tool_version`
- `data_type` -> `compress_to_fp16` for `RVC2` and `RVC3`, `quantization_mode` for `RVC4`
- `shaves` -> `number_of_shaves`
- `optimizer_params` -> `mo_args`
- `compile_params` -> `compile_tool_args`
