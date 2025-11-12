import shutil
from pathlib import Path
import os
from hubai_sdk import HubAIClient
import pytest

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def test_rvc4_int8_conversion(client: HubAIClient, base_model_path: str):
    response = client.convert.RVC4(
        path=base_model_path,
        name="test-sdk-conversion-rvc4-int8",
        target_precision="INT8",
        quantization_data="GENERAL",
    )

    assert response is not None
    downlaoded_path = response.downloaded_path

    downlaoded_path = downlaoded_path.resolve()

    assert Path.exists(downlaoded_path)
    shutil.rmtree(str(downlaoded_path.parent))

    client.models.delete_model("test-sdk-conversion-rvc4-int8")
