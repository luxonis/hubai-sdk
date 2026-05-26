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

## Python Setup

```python
import os
from hubai_sdk import HubAIClient

client = HubAIClient(api_key=os.getenv("HUBAI_API_KEY"))
```

## Minimal RVC2 Conversion

```python
response = client.convert.RVC2(
    path="/path/to/model.onnx",
    name="my-rvc2-model",
)

print(response.downloaded_path)
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

When `.xml` and `.bin` live beside each other, passing the XML path is enough:

```python
response = client.convert.RVC2(
    path="/path/to/model.xml",
    name="openvino-rvc2-model",
)
```

If the `.bin` file lives elsewhere, pass it through `opts`:

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

Pass the zip path itself. Do not pass `CUSTOM` by itself.

```python
response = client.convert.RVC4(
    path="/path/to/model.onnx",
    name="my-rvc4-model",
    quantization_mode="INT8_STANDARD",
    quantization_data="/path/to/quantization.zip",
    max_quantization_images=50,
)
```

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

hubai convert RVC2 --path /path/to/model.onnx --name my-rvc2-legacy-model --superblob false

hubai convert RVC4 --path /path/to/model.onnx --name my-rvc4-model --quantization-mode INT8_STANDARD --quantization-data GENERAL --max-quantization-images 50

hubai convert Hailo --path /path/to/model.onnx --name my-hailo-model
```

Use `hubai convert --help` and `hubai convert <TARGET> --help` when you need the current CLI flag spellings inside a live environment.
