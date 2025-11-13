import shutil
from pathlib import Path

from hubai_sdk import HubAIClient
import pytest
import os
import uuid

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def test_rvc2_legacy_conversion(client: HubAIClient, base_model_path: str):
    model_name = f"test-sdk-conversion-rvc2-legacy-{str(uuid.uuid4())}"
    response = client.convert.RVC2(
        path=base_model_path,
        name=model_name,
        superblob=False,
    )

    assert response is not None
    downlaoded_path = response.downloaded_path

    downlaoded_path = downlaoded_path.resolve()

    assert Path.exists(downlaoded_path)
    shutil.rmtree(str(downlaoded_path.parent))

    client.models.delete_model(str(response.instance.model_id))
