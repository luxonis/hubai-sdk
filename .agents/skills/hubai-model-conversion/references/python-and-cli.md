# Python and CLI

## Authentication

Prefer environment-based auth in automation:

```bash
export HUBAI_API_KEY="your-api-key"
```

For interactive local work:

```bash
hubai login
```

`HubAIClient()` resolves credentials in this order: explicit `api_key=`, `HUBAI_API_KEY`, then stored credentials from `hubai login`.

## Python Setup

```python
from hubai_sdk import HubAIClient

client = HubAIClient()
```

For automation or explicit overrides:

```python
import os
from hubai_sdk import HubAIClient

client = HubAIClient(api_key=os.getenv("HUBAI_API_KEY"))
```

`HubAIClient(...)` verifies the resolved API key on initialization, so bad auth fails early.

## Minimal RVC2 Conversion

```python
response = client.convert.RVC2(
    path="/path/to/model.onnx",
    name="my-rvc2-model",
)

print(response.downloaded_path)
```

## Download Location

`output_dir` is a directory path. If you omit it, the downloader creates a directory named from the exported instance slug in the current working directory.

```python
response = client.convert.RVC2(
    path="/path/to/model.onnx",
    name="my-rvc2-model",
    output_dir="/tmp/my-export",
)

print(response.downloaded_path)
```

## YAML Config or NN Archive Input

The same `path` parameter can point to a model file, a YAML config, or an NN Archive:

```python
response = client.convert.RVC4(
    path="/path/to/model.yaml",
    name="my-rvc4-model",
    quantization_mode="FP16_STANDARD",
)

response = client.convert.RVC4(
    path="/path/to/model.tar.xz",
    name="my-rvc4-model",
    quantization_mode="FP16_STANDARD",
)
```

## Legacy RVC2 Blob Export

Use this when the consumer needs the legacy blob format instead of a superblob:

```python
response = client.convert.RVC2(
    path="/path/to/model.onnx",
    name="my-rvc2-legacy-model",
    superblob=False,
)
```

## OpenVINO IR Conversion

When `.xml` and `.bin` live beside each other with the same basename, passing the XML path is enough:

```python
response = client.convert.RVC2(
    path="/path/to/model.xml",
    name="openvino-rvc2-model",
)
```

Equivalent:

```python
response = client.convert.RVC2(
    path="/path/to/model.bin",
    name="openvino-rvc2-model",
)
```

If the `.bin` file lives elsewhere, pass it explicitly through `opts`:

```python
response = client.convert.RVC2(
    path="/path/to/model.xml",
    name="openvino-rvc2-model",
    opts={"input_bin": "/other/path/model.bin"},
)
```

## RVC4 With Hosted Quantization

```python
response = client.convert.RVC4(
    path="/path/to/model.onnx",
    name="my-rvc4-model",
    quantization_mode="INT8_STANDARD",
    quantization_data="GENERAL",
    max_quantization_images=50,
)
```

## RVC4 With Custom Calibration Zip

Pass the local existing zip path itself. Do not pass `CUSTOM` by itself; the SDK maps the zip path to `CUSTOM` automatically.

```python
response = client.convert.RVC4(
    path="/path/to/model.onnx",
    name="my-rvc4-model",
    quantization_mode="INT8_STANDARD",
    quantization_data="/path/to/quantization.zip",
    max_quantization_images=50,
)
```

## PyTorch YOLO Input

PyTorch inputs are supported only for YOLO models. Normally rely on auto-detection:

```python
response = client.convert.RVC4(
    path="/path/to/model.pt",
    name="my-yolo-model",
    quantization_mode="FP16_STANDARD",
)
```

Only try explicit `yolo_version` if auto-detection fails and you still need to attempt the conversion.

## Non-YOLO PyTorch Fallback

Preferred recovery path:

1. If the user already has ONNX, TFLite, OpenVINO IR, or NN Archive, use that instead.
2. Otherwise, if the required local framework dependencies are available and the workflow allows a small local prep step, load the checkpoint locally and export it to ONNX first.
3. Then continue with the normal hosted conversion flow using the exported ONNX, TFLite, IR, or NN Archive artifact.

## Cleanup of Created Hub Resources

If the conversion created a fresh model hierarchy and you only want the downloaded files, prefer deleting the created model after download:

```python
client.models.delete_model(response.instance.model_id)
```

Equivalent CLI:

```bash
hubai model delete <model_id>
```

Do not delete `model_id`, `variant_id`, or other Hub resources that were supplied up front instead of being created by the current conversion flow.

## Generic Entry Point

Use this only when the target is dynamic:

```python
from hubai_sdk.utils.types import Target

response = client.convert.convert(
    target=Target.RVC2,
    path="/path/to/model.onnx",
    name="converted-model",
)
```

## CLI Examples

```bash
hubai convert RVC2 --path /path/to/model.onnx --name my-rvc2-model

hubai convert RVC2 rvc2.superblob False --path /path/to/model.onnx --name my-rvc2-legacy-model

hubai convert RVC4 --path /path/to/model.onnx --name my-rvc4-model --quantization-mode INT8_STANDARD --quantization-data GENERAL --max-quantization-images 50

hubai convert Hailo --path /path/to/model.onnx --name my-hailo-model
```

For extra target-specific settings, pass positional `key value` pairs after the target, for example `rvc2.superblob False`.

For extra non-target config settings, use the same positional form, for example `input_bin /other/path/model.bin`.

Use `hubai convert --help` or `hubai convert <TARGET> --help` when you need the current CLI spelling inside a live environment.
