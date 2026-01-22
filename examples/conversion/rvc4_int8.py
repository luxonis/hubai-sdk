import os

from hubai_sdk import HubAIClient

from pathlib import Path
import argparse

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
    name="test-sdk-conversion-rvc4-int8",
    quantization_mode="INT8_STANDARD",
    quantization_data="GENERAL",
    max_quantization_images=100,
)

# Extract the model instance
model = response.instance

print(f"Model instance: {model}\n")

# Extract the path to the downloaded model
downlaoded_path = response.downloaded_path
downlaoded_path = downlaoded_path.resolve()

assert Path.exists(downlaoded_path)
print(f"Model downloaded to: {downlaoded_path}\n")

# Delete the model
client.models.delete_model("test-sdk-conversion-rvc4-int8")
