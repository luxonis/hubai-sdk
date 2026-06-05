---
name: hubai-model-conversion
description: Convert models with HubAI SDK or `hubai convert` for RVC2, RVC3, RVC4, or Hailo using HubAI's hosted export flow. Use for hosted conversion requests, including replacements for `blobconverter` and legacy tools.luxonis.com YOLO-conversion workflows.
---

# HubAI Model Conversion

Use this skill for HubAI's hosted model conversion flow through the Python SDK or `hubai convert`.

## Use This Skill When

- Converting models for `RVC2`, `RVC3`, `RVC4`, or `Hailo`
- Using `HubAIClient(...).convert.<TARGET>(...)` or `hubai convert <TARGET> ...`
- Replacing older `blobconverter` or tools.luxonis.com YOLO-conversion workflows

Do not use this skill for fully local or offline conversion, or when the source artifact must not be uploaded to Hub. For that, use `luxonis/modelconverter`.

## Inputs To Confirm

- source artifact path
- export target
- working authentication
- whether the task needs SDK code, a CLI command, or an executed conversion
- any target-specific options or quantization requirements
- desired output handling and whether Hub resources should be kept or cleaned up
- optional Hub metadata only when the caller cares about resource naming, versioning, tags, visibility, or repository links

## Workflow

1. Confirm this is a hosted conversion task, not a local-only conversion request.
2. Identify the source artifact, export target, and whether the task wants SDK code, CLI code, or a finished conversion run.
3. If the task is wiring Python code and the project does not already depend on `hubai-sdk`, add it before wiring conversion.
4. Prefer the target helper `client.convert.<TARGET>(...)` or `hubai convert <TARGET> --path ...`.
5. Read [references/python-and-cli.md](references/python-and-cli.md) for authentication, SDK setup, CLI syntax, and runnable examples.
6. Read [references/targets-and-constraints.md](references/targets-and-constraints.md) for supported inputs, target knobs, Hub parameters, quantization rules, and `blobconverter` migration mapping.
7. Return the converted artifact path, runnable code or command, or a migration diff.
8. If cleanup is requested, delete only resources created by the current conversion flow.

## Critical Guardrails

- Hosted conversion uploads the source artifact to Hub and supports only single-stage models.
- If the user wants local-only conversion or no Hub upload, stop and switch to `modelconverter`.
- PyTorch `.pt` and `.pth` inputs are supported only for YOLO models. Rely on auto-detected `yolo_version` unless detection fails.
- For RVC4 custom calibration, pass a real local `.zip` path. Do not pass `CUSTOM`.
- Never delete Hub resources that existed before the current conversion flow or were passed in by ID.

## Validation

The task is complete when:

- a conversion run finishes successfully when execution is part of the task
- the converted artifact exists at `response.downloaded_path` or the CLI download path when execution is part of the task
- the exported target matches the requested platform
- custom calibration flows use a real `.zip` path and that upload succeeds when applicable
- if cleanup was requested, only the Hub resources created in this flow are deleted

## Expected Output

Finish with one of these:

- a converted model with its resulting artifact path,
- runnable Python conversion code,
- a working CLI command or script,
- or a precise migration diff from an older `blobconverter` flow.

Call out missing prerequisites explicitly: source path, target selection, authentication, target-specific options, custom quantization zip, or the `hubai-sdk` dependency itself.
