import shutil
from pathlib import Path

from hubai_sdk import HubAIClient
import pytest
import os

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def test_rvc2_legacy_conversion(client: HubAIClient, base_model_path: str):
    response = client.convert.RVC2(
        path=base_model_path,
        name="test-sdk-conversion-rvc2-legacy",
        superblob=False,
    )

    assert response is not None
    downlaoded_path = response.downloaded_path

    downlaoded_path = downlaoded_path.resolve()

    assert Path.exists(downlaoded_path)
    shutil.rmtree(str(downlaoded_path.parent))

    client.models.delete_model("test-sdk-conversion-rvc2-legacy")
