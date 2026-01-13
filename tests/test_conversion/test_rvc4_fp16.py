import shutil
from pathlib import Path

from hubai_sdk import HubAIClient
import pytest
import os
import uuid
os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def test_rvc4_fp16_conversion(client: HubAIClient, base_model_path: str):
    model_name = f"test-sdk-conversion-rvc4-fp16-{str(uuid.uuid4())}"
    response = client.convert.RVC4(
        path=base_model_path,
        name=model_name,
        quantization_mode="FP16_STANDARD",
    )

    assert response is not None
    downlaoded_path = response.downloaded_path

    downlaoded_path = downlaoded_path.resolve()

    assert Path.exists(downlaoded_path)
    shutil.rmtree(str(downlaoded_path.parent))

    client.models.delete_model(str(response.instance.model_id))
