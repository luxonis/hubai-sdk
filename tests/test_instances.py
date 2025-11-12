import shutil
import os
import pytest
from loguru import logger

from hubai_sdk import HubAIClient
from hubai_sdk.utils.types import ModelType

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"

def test_list_instances(client: HubAIClient):
    """Test listing instances functionality."""
    instances = client.instances.list_instances()
    assert instances is not None
    logger.info(f"Found {len(instances)} instances")
    assert isinstance(instances, list)
    assert len(instances) >= 0


def test_get_instance(client: HubAIClient):
    """Test getting a specific instance."""
    instances = client.instances.list_instances()
    if not instances:
        pytest.skip("No instances available to test get_instance")
    instance = client.instances.get_instance(instances[0].id)
    assert instance is not None
    assert hasattr(instance, "id")
    assert str(instance.id) == str(instances[0].id)


def test_create_and_delete_instance(client: HubAIClient):
    """Test creating and deleting an instance."""
    variants = client.variants.list_variants()
    if not variants:
        pytest.skip("No variants available to test create_and_delete_instance")
    instance = client.instances.create_instance(
        name="test-sdk-instance-pytest",
        variant_id=variants[0].id,
        model_type=ModelType.ONNX,
    )
    assert instance is not None

    client.instances.delete_instance(instance.id)


def test_e2e_instance(client: HubAIClient, model_base_path: str):
    """Test end-to-end instance functionality."""
    variants = client.variants.list_variants()
    if not variants:
        pytest.skip("No variants available to test e2e_instance")
    instance = client.instances.create_instance(
        name="test-sdk-instance-base",
        variant_id=variants[0].id,
        model_type=ModelType.ONNX,
        input_shape=[1, 3, 288, 512],
    )
    assert instance is not None

    client.instances.upload_file(
        model_base_path, instance.id
    )

    downloaded_path = client.instances.download_instance(instance.id)

    client.instances.delete_instance(instance.id)

    # Removing downloaded files, combines downlaoded path and pwd of the scirpt
    downloaded_path = downloaded_path.resolve()
    shutil.rmtree(str(downloaded_path.parent))
