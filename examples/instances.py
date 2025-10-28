import os
import shutil

from hubai_sdk import HubAIClient
from hubai_sdk.services import variants
from hubai_sdk.utils.types import ModelType

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

# Get variants
variants = client.variants.list_variants()

# Create instance
instance = client.instances.create_instance(
    name="test-sdk-instance-base",
    variant_id=variants[0].id,
    model_type=ModelType.ONNX,
    input_shape=[1, 3, 288, 512],
)
assert instance is not None

# Upload base model file to the instance
client.instances.upload_file(
    args.file, instance.id
)

# Get config of the instance
config = client.instances.get_config(instance.id)
print(f"Config: {config}\n")

# Get files of the instance
files = client.instances.get_files(instance.id)
print(f"Files: {files}\n")

# Download instance
downloaded_path = client.instances.download_instance(instance.id)
print(f"Instance downloaded to: {downloaded_path}\n")

# Delete instance
client.instances.delete_instance(instance.id)

# Removing downloaded files, combines downlaoded path and pwd of the scirpt
downloaded_path = downloaded_path.resolve()
shutil.rmtree(str(downloaded_path.parent))
