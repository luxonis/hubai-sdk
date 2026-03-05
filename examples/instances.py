import argparse
import os
import shutil

from loguru import logger

from hubai_sdk import HubAIClient
from hubai_sdk.utils.types import ModelType

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

# Upload base model file to the instance
client.instances.upload_file(args.file, instance.id)

# Get config of the instance
config = client.instances.get_config(instance.id)
logger.info(f"Config: {config}")

# Get files of the instance
files = client.instances.get_files(instance.id)
logger.info(f"Files: {files}")

# Download instance
downloaded_path = client.instances.download_instance(instance.id)
logger.info(f"Instance downloaded to: {downloaded_path}")

# Delete instance
client.instances.delete_instance(instance.id)

# Removing downloaded files, combines downlaoded path and pwd of the scirpt
downloaded_path = downloaded_path.resolve()
shutil.rmtree(str(downloaded_path.parent))
