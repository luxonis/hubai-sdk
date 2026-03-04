import os
import shutil
import uuid
from pathlib import Path

from hubai_sdk import HubAIClient
from hubai_sdk.utils.types import ModelType

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def test_list_instances(client: HubAIClient):
    """Test listing instances functionality."""
    instances = client.instances.list_instances()
    assert instances is not None
    assert isinstance(instances, list)
    assert len(instances) >= 0


def test_get_instance(client: HubAIClient, test_instance_id: str):
    """Test getting a specific instance."""
    instance = client.instances.get_instance(test_instance_id)
    assert instance is not None
    assert hasattr(instance, "id")
    assert str(instance.id) == str(test_instance_id)


def test_create_and_delete_instance(client: HubAIClient, test_variant_id: str):
    """Test creating and deleting an instance."""
    instance_name = f"test-sdk-instance-pytest-{uuid.uuid4()!s}"
    instance = client.instances.create_instance(
        name=instance_name,
        variant_id=test_variant_id,
        model_type=ModelType.ONNX,
    )
    assert instance is not None
    assert instance.name == instance_name

    client.instances.delete_instance(instance.id)


def test_e2e_instance(
    client: HubAIClient, base_model_path: str, test_variant_id: str
):
    """Test end-to-end instance functionality."""
    instance_name = f"test-sdk-instance-base-{uuid.uuid4()!s}"
    instance = client.instances.create_instance(
        name=instance_name,
        variant_id=test_variant_id,
        model_type=ModelType.ONNX,
        input_shape=[1, 3, 288, 512],
    )
    assert instance is not None
    assert instance.name == instance_name

    client.instances.upload_file(base_model_path, instance.id)

    downloaded_path = client.instances.download_instance(instance.id)

    assert Path(downloaded_path).exists()

    client.instances.delete_instance(instance.id)

    # Removing downloaded files, combines downlaoded path and pwd of the scirpt
    downloaded_path = downloaded_path.resolve()
    shutil.rmtree(str(downloaded_path.parent))
