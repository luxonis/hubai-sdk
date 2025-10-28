import os

from hubai_sdk import HubAIClient

from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to the PyTorch YOLO model file (.pt)")

args = parser.parse_args()

model_path = args.model_path

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

# Convert model to RVC4
response = client.convert.RVC4(
    path=model_path,
    name="test-sdk-conversion-yolo-v8",
    target_precision="INT8",
    quantization_data="GENERAL",
    max_quantization_images=100,
    yolo_input_shape=[512, 288],
    yolo_version="yolov8",
)

# Extract the model instance
model = response.instance

print(f"Model instance: {model}\n")
print(f"Detected YOLO version: {model.yolo_version}\n")

# Extract the path to the downloaded model
downlaoded_path = response.downloaded_path
downlaoded_path = downlaoded_path.resolve()

assert Path.exists(downlaoded_path)
print(f"Model downloaded to: {downlaoded_path}\n")

# Delete the model
client.models.delete_model("test-sdk-conversion-yolo-v8")
