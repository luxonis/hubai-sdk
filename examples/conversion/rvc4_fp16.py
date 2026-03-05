import argparse
import os
from pathlib import Path

from loguru import logger

from hubai_sdk import HubAIClient

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", "-m", type=str, required=True)
args = parser.parse_args()

model_path = args.model_path

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

# Convert model to RVC4
response = client.convert.RVC4(
    path=model_path,
    name="test-sdk-conversion-rvc4-fp16",
    quantization_mode="FP16_STANDARD",
)

# Extract the model instance
model = response.instance

logger.info(f"Model instance: {model}")

# Extract the path to the downloaded model
downloaded_path = response.downloaded_path.resolve()

assert Path.exists(downloaded_path)
logger.info(f"Model downloaded to: {downloaded_path}")

# Delete the model
client.models.delete_model("test-sdk-conversion-rvc4-fp16")
