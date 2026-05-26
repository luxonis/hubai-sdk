---
name: hubai-model-conversion
description: Convert models with HubAI SDK or `hubai convert` to RVC2, RVC3, RVC4, or Hailo. Use when the user asks to export ONNX, TFLite, OpenVINO IR, NN Archive, or YOLO models, or mentions quantization, calibration zip files, or `blobconverter` migration.
---

# HubAI Model Conversion

Use this skill for HubAI's hosted model conversion flow.

## Use This Skill When

- Converting models for `RVC2`, `RVC3`, `RVC4`, or `Hailo`
- Using `HubAIClient(...).convert.<TARGET>(...)` or `hubai convert <TARGET> ...` to run hosted conversion
- Migrating older `blobconverter` usage to HubAI SDK
- Handling RVC4 quantization, dataset IDs, or custom calibration `.zip` files
- Working with ONNX, TFLite, OpenVINO IR (`.xml` + `.bin`), NN Archive tarballs, YAML configs, or PyTorch YOLO weights

Do not use this skill for fully local or offline conversion. For that, use `luxonis/modelconverter`.

## Inputs Needed

- input model path
- export target: `RVC2`, `RVC3`, `RVC4`, or `Hailo`
- authentication via `HUBAI_API_KEY` or a working `hubai login`
- `yolo_version` for PyTorch YOLO inputs (`.pt`, `.pth`)
- quantization source for RVC4 INT8 flows when the default is not acceptable
- desired output handling when the converted artifact needs to be stored, returned, or passed downstream

## Quick Checks

1. Confirm whether the task needs Python SDK code, a CLI command, or both.
2. Confirm authentication: `HUBAI_API_KEY` or `hubai login`.
3. Identify the input artifact and the export target.
4. Prefer the target-specific helper (`RVC2`, `RVC3`, `RVC4`, `Hailo`) over the generic `convert(target=...)` entrypoint unless the caller needs to select the target programmatically. The target must always be specified.

## Workflow

1. If the project does not already depend on `hubai-sdk`, add it before wiring conversion.
2. For application code, use `HubAIClient(...).convert.<TARGET>(...)`.
3. For scripts, CI, or operator workflows, use `hubai convert <TARGET> --path ...`.
4. Put target-specific knobs on the target helper first. Use `opts` only for settings that do not have a direct parameter.
5. Return or store `response.downloaded_path` when downstream code needs the converted artifact.
6. Preserve Hub metadata only when the task needs it: model name, tasks, variant version, tags, links, repository URL, commit hash, visibility, and deployability.

## Guardrails

- Online conversion supports only single-stage models.
- PyTorch inputs (`.pt`, `.pth`) require `yolo_version`.
- RVC2 and RVC3 ignore `quantization_mode`, `quantization_data`, and `max_quantization_images`.
- Legacy RVC2 blob export requires `superblob=False`.
- RVC4 defaults calibration data to `RANDOM` unless the mode is `FP16_STANDARD` or `FP32_STANDARD`, or a custom source is supplied.
- Custom quantization data must be passed as a real `.zip` path. Passing `CUSTOM` alone is invalid.
- OpenVINO IR can use only the `.xml` path when the `.bin` file is beside it; otherwise pass the bin path through `opts`.
- If the user wants local-only conversion or no Hub upload, stop and switch to `modelconverter`.

## Validation

The task is complete when:

- the Hub export job completes successfully
- the converted artifact exists at `response.downloaded_path` or the CLI download path
- the exported target matches the requested platform
- PyTorch YOLO inputs include a valid `yolo_version`
- custom calibration flows use a real `.zip` path and that upload succeeds

## Common Failure Modes

- `HUBAI_API_KEY` is missing: export it explicitly in the shell before running conversion
- keyring or desktop secret storage is unavailable: prefer `HUBAI_API_KEY` over stored login
- `.pt` or `.pth` input without `yolo_version`: conversion fails for PyTorch YOLO models
- custom quantization input is passed as `CUSTOM` instead of a `.zip` path: invalid, pass the file path directly
- quantization path does not end in `.zip`: invalid custom calibration input
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
