---
name: hubai-model-conversion
description: Convert models with HubAI SDK or `hubai convert` for RVC2, RVC3, RVC4, or Hailo using HubAI's hosted export flow. Use for hosted conversion requests, including replacements for `blobconverter` and legacy tools.luxonis.com YOLO-conversion workflows.
---

# HubAI Model Conversion

Use this skill for HubAI's hosted model conversion flow.

## Use This Skill When

- Converting models for `RVC2`, `RVC3`, `RVC4`, or `Hailo`
- Using `HubAIClient(...).convert.<TARGET>(...)` or `hubai convert <TARGET> ...` to run hosted conversion
- Replacing older `blobconverter` or tools.luxonis.com YOLO-conversion workflows
- Handling conversion requests that involve RVC4 quantization datasets or custom calibration `.zip` files
- Working with ONNX, TFLite, OpenVINO IR (`.xml` + `.bin`), PyTorch YOLO weights, YAML config inputs, or NN Archive inputs

Do not use this skill for fully local or offline conversion. For that, use `luxonis/modelconverter`.

## Inputs Needed

- conversion input `path`: model file, YAML config, or NN Archive
- export target: `RVC2`, `RVC3`, `RVC4`, or `Hailo`
- authentication via `HUBAI_API_KEY` or a working `hubai login`; `HubAIClient()` can use stored login state, but `HUBAI_API_KEY` is preferred in automation
- for PyTorch inputs (`.pt`, `.pth`): confirm the model is YOLO and rely on auto-detection by default; `yolo_input_shape` defaults to `[640, 640]`
- quantization source for RVC4 calibration-based modes only when the caller wants something other than the default dataset: predefined dataset, dataset ID (`aid_...`), or a local `.zip`
- desired output handling when the converted artifact needs to be stored, returned, or passed downstream
- whether to keep the created Hub resources or clean them up after download
- optional Hub metadata only if the caller cares about resource naming, versioning, tags, visibility, or repository links

## Quick Checks

1. Confirm whether the task needs Python SDK code, a CLI command, or both.
2. Confirm authentication: explicit `HUBAI_API_KEY` or a working `hubai login`. `HubAIClient()` can use stored login state when no key is passed.
3. Identify the input artifact and the export target.
4. If the input is `.pt` or `.pth`, first decide whether it is a YOLO checkpoint. Rely on auto-detected `yolo_version` by default for YOLO, and only fall back to explicit `yolo_version` if detection fails.
5. Prefer the target-specific helper (`RVC2`, `RVC3`, `RVC4`, `Hailo`) over the generic `convert(target=...)` entrypoint unless the caller needs to select the target programmatically. The target must always be specified.
6. If using the CLI with extra config settings, remember that `hubai convert` takes positional `key value` pairs such as `input_bin /other/path/model.bin` or `rvc2.superblob False`.
7. Decide whether the created Hub resources should be kept or cleaned up after download. Never delete pre-existing Hub resources that were passed in by ID.

## Workflow

01. If the project does not already depend on `hubai-sdk`, add it before wiring conversion.
02. For application code, use `HubAIClient(...).convert.<TARGET>(...)`.
03. For scripts, CI, or operator workflows, use `hubai convert <TARGET> --path ...`. When extra config settings are needed, pass positional key/value pairs such as `hubai convert RVC2 input_bin /other/path/model.bin --path /path/to/model.xml` or `hubai convert RVC2 rvc2.superblob False --path ...`.
04. Put target-specific knobs on the target helper first. Use `opts` only for settings that do not have a direct parameter. On the CLI, generic Hub fields stay as flags and extra config settings stay as positional pairs.
05. Return or store `response.downloaded_path` when downstream code needs the converted artifact.
06. Treat `output_dir` as a directory path. If it is omitted, the downloader creates a directory named from the exported instance slug under the current working directory.
07. Preserve Hub metadata only when the task needs it: model name, tasks, variant version, tags, links, repository URL, commit hash, visibility, and deployability.
08. Remember that hosted conversion uploads the source artifact to Hub and downloads the exported instance when the job completes.
09. If the user only wants the local output files, decide whether to clean up the Hub resources created by the conversion. Delete only resources created in this flow.
10. When the conversion created a fresh model hierarchy and cleanup is requested, prefer deleting the created model via `client.models.delete_model(response.instance.model_id)` or `hubai model delete <model_id>`.

## Guardrails

- Online conversion supports only single-stage models.
- Hosted conversion uploads the source artifact to Hub. If the user wants local-only conversion or no Hub upload, stop and switch to `modelconverter`.
- PyTorch inputs (`.pt`, `.pth`) are supported only for YOLO models. Normally do not pass `yolo_version`; rely on auto-detection. `yolo_input_shape` defaults to `[640, 640]`.
- RVC2 and RVC3 ignore `quantization_mode`, `quantization_data`, and `max_quantization_images`.
- Legacy RVC2 blob export requires `superblob=False`.
- RVC4 only: on calibration-based modes, omitted `quantization_data` defaults to `RANDOM`. `FP16_STANDARD` does not use calibration data.
- `FP32_STANDARD` may appear in runtime types, but do not suggest it, document it as a supported mode, or use it in conversions.
- For a local calibration `.zip`, pass the zip path itself. Passing `CUSTOM` directly is invalid; the SDK normalizes the zip path to `CUSTOM` automatically.
- OpenVINO IR can use just the `.xml` path when the `.bin` sibling is beside it. If the `.bin` lives elsewhere, pass it explicitly through `opts={"input_bin": ...}` in Python or `input_bin /path/to/model.bin` on the CLI.
- Cleanup is optional. Never delete Hub models, variants, or instances that existed before the current conversion flow.

## Validation

The task is complete when:

- the Hub export job completes successfully
- the converted artifact exists at `response.downloaded_path` or the CLI download path
- the exported target matches the requested platform
- custom calibration flows use a real `.zip` path and that upload succeeds
- if cleanup was requested, only the Hub resources created in this flow are deleted

## Common Failure Modes

- `HUBAI_API_KEY` is missing: export it explicitly in the shell before running conversion
- keyring or desktop secret storage is unavailable: prefer `HUBAI_API_KEY` over stored login
- `.pt` or `.pth` input is a YOLO model the API does not support, or auto-detection fails: try explicit `yolo_version` as a fallback
- `.pt` or `.pth` input is a non-YOLO checkpoint: obtain or locally export a supported interchange artifact such as ONNX before continuing with hosted conversion
- custom quantization input is passed as `CUSTOM`, a non-zip path, or a missing file: invalid, pass a real local `.zip`
- OpenVINO IR `.xml` is missing its sibling `.bin`, or the explicit `input_bin` path is missing
- CLI extra target-specific settings are not passed as key/value pairs: the CLI rejects an odd number of positional setting tokens
- multi-stage config or archive input: hosted conversion supports only single-stage models

## References

- Read [references/python-and-cli.md](references/python-and-cli.md) for working SDK and CLI examples.
- Read [references/targets-and-constraints.md](references/targets-and-constraints.md) when choosing target-specific options, calibration inputs, or migration behavior.

## Expected Output

Finish with one of these:

- a converted model with its resulting artifact path,
- runnable Python conversion code,
- a working CLI command or script,
- or a precise migration diff from an older `blobconverter` flow.

Call out missing prerequisites explicitly: API key, input artifact path, target selection, custom quantization zip, or the `hubai-sdk` dependency itself.
