# HubAI SDK

Python SDK for interacting with Luxonis HubAI - a platform for managing, converting, and deploying machine learning models for Luxonis OAK devices.

## ✨ Features

- **Model Management**: Create, list, update, and delete HubAI models
- **Variant Management**: Manage HubAI model variants and versions
- **Instance Management**: Create and manage HubAI model instances
- **Model Conversion**: Convert HubAI models to various formats including:
  - RVC2
  - RVC3
  - RVC4
  - Hailo
- **CLI Tools**: Command-line interface for all operations
- **Type Safety**: Full type hints for better developer experience

## 📦 Installation

Install the package using pip:

```bash
pip install hubai-sdk
```

Or install from source:

```bash
git clone https://github.com/luxonis/hubai-sdk.git
cd hubai-sdk
pip install -e .
```

## 📋 Requirements

- Python 3.10 or higher
- Valid Luxonis HubAI API key - you can get it from [HubAI Team Settings](https://hub.luxonis.com/team-settings)

## 🔐 Authentication

### Get Your API Key

1. Visit [HubAI Team Settings](https://hub.luxonis.com/team-settings)
1. Generate or copy your API key

### Set API Key

You can authenticate in several ways:

**Option 1: Environment Variable**

```bash
export HUBAI_API_KEY="your-api-key-here"
```

This will store the API key in your environment variable and will be used by the SDK automatically. It is valid for the current session only.

**Option 2: CLI Login**

```bash
hubai login
```

This will open a browser to generate a new API key and prompt you to enter it, which will be securely stored. Use `hubai --relogin` to relogin with different API key or `hubai logout` to logout.

**Option 3: Pass API Key Directly**

```python
from hubai_sdk import HubAIClient

client = HubAIClient(api_key="your-api-key-here")
```

## 🚀 Quick Start

### Python SDK Usage

```python
import os
from hubai_sdk import HubAIClient

# Initialize client
api_key = os.getenv("HUBAI_API_KEY")
client = HubAIClient(api_key=api_key)

# List all models
models = client.models.list_models()
print(f"Found {len(models)} models")

# Get a specific model
model = client.models.get_model(models[0].id)
print(f"Model: {model.name}")

# Convert a model to RVC2 format
response = client.convert.RVC2(
    path="/path/to/your/model.onnx",
    name="my-converted-model"
)

print(f"Converted model downloaded to: {response.downloaded_path}")
```

## 🛠️ Services

The SDK provides four main services accessible through the `HubAIClient`:

### 🤖 Models Service (`client.models`)

Manage ML models in HubAI.

```python
# List models
models = client.models.list_models(
    tasks=["OBJECT_DETECTION"],
    is_public=True,
    limit=10
)

# Get model by ID or slug
model = client.models.get_model("model-id-or-slug")

# Create a new model
new_model = client.models.create_model(
    name="my-model",
    license_type="MIT",
    is_public=False,
    description="My awesome model",
    tasks=["OBJECT_DETECTION"]
)

# Update a model
updated_model = client.models.update_model(
    model_id,
    license_type="Apache 2.0",
    description="Updated description"
)

# Delete a model
client.models.delete_model(model_id)
```

### 🔄 Variants Service (`client.variants`)

Manage model variants and versions.

```python
# List variants (optionally filtered by model)
variants = client.variants.list_variants(model_id="model-id")

# Get variant by ID
variant = client.variants.get_variant("variant-id")

# Create a new variant
new_variant = client.variants.create_variant(
    name="my-variant",
    model_id="model-id",
    variant_version="1.0.0",
    description="First version"
)

# Delete a variant
client.variants.delete_variant("variant-id")
```

### 📦 Instances Service (`client.instances`)

Manage model instances (specific configurations of variants).

```python
# Create an instance
instance = client.instances.create_instance(
    name="my-instance",
    variant_id="variant-id",
    model_type=ModelType.ONNX,
    input_shape=[1, 3, 288, 512]
)

# Upload a file to instance
client.instances.upload_file("/path/to/nn_archive.tar.xz", instance.id)

# Get instance config
config = client.instances.get_config(instance.id)

# Download instance
downloaded_path = client.instances.download_instance(instance.id)

# Delete instance
client.instances.delete_instance(instance.id)
```

### ⚡ Conversion Service (`client.convert`)

Convert models to various formats.

#### RVC2 Conversion

Convert models for Luxonis OAK devices:

```python
response = client.convert.RVC2(
    path="/path/to/model.onnx",
    name="converted-model",
    compress_to_fp16=True,
    number_of_shaves=8,
    superblob=True
)
```

#### RVC4 Conversion

Convert models to Qualcomm SNPE format:

```python
response = client.convert.RVC4(
    path="/path/to/model.onnx",
    name="converted-model",
    target_precision="INT8",
    use_per_channel_quantization=True,
    htp_socs=["sm8550"]
)
```

#### Generic Conversion

Convert to any supported target:

```python
from hubai_sdk.utils.types import Target

response = client.convert.convert(
    target=Target.RVC2,  # or Target.RVC4, Target.HAILO, etc.
    path="/path/to/model.onnx",
    name="converted-model",
    target_precision="INT8",
    input_shape=[1, 3, 288, 512]
)
```

## 💻 CLI Usage

The SDK also provides a command-line interface:

```bash
# Login
hubai login

# List models
hubai model ls

# Get model info
hubai model info <model-id-or-slug>

# Create a model
hubai model create "my-model" --license-type MIT --tasks OBJECT_DETECTION

# Convert a model
hubai convert RVC2 --path /path/to/model.onnx --name "my-model"

# List variants
hubai variant ls

# List instances
hubai instance ls
```

For more CLI options, use the `--help` flag:

```bash
hubai --help
hubai model --help
hubai convert --help
```

## 📚 Examples

See the `examples/` directory for more detailed usage examples:

- **`examples/models.py`**: Model management operations
- **`examples/variants.py`**: Variant management operations
- **`examples/instances.py`**: Instance management and file operations
- **`examples/conversion/`**: Model conversion examples for different formats

### Key Types

- **`License`**: Model license types (MIT, Apache 2.0, etc.)
- **`Task`**: ML tasks (OBJECT_DETECTION, CLASSIFICATION, etc.)
- **`Target`**: Conversion targets (RVC2, RVC4, HAILO, etc.)
- **`TargetPrecision`**: Model precision (INT8, FP16, FP32)
- **`ModelType`**: Model file formats (ONNX, OPENVINO, etc.)

## 🔨 Development

### Setup Development Environment

```bash
git clone https://github.com/luxonis/hubai-sdk.git
cd hubai-sdk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/luxonis/hubai-sdk/issues)
- **Email**: support@luxonis.com
- **Documentation**: [HubAI Platform](https://hub.luxonis.com)

## 📄 License

This project is licensed under the terms specified in the project repository.

## 🔗 Links

- **Repository**: [https://github.com/luxonis/hubai-sdk](https://github.com/luxonis/hubai-sdk)
- **HubAI Platform**: [https://hub.luxonis.com](https://hub.luxonis.com)
- **Luxonis**: [https://luxonis.com](https://luxonis.com)
