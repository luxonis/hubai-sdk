import pytest
from loguru import logger
import os
import uuid
from hubai_sdk import HubAIClient

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"

def test_list_models(client: HubAIClient):
    """Test listing models functionality."""
    models = client.models.list_models()
    assert models is not None
    assert isinstance(models, list)


def test_get_model(client: HubAIClient, test_model_id: str):
    """Test getting a specific model."""
    model = client.models.get_model(test_model_id)
    assert model is not None
    assert hasattr(model, "id")
    assert str(model.id) == str(test_model_id)


def test_create_and_delete_model(client: HubAIClient):
    """Test creating and deleting a model."""
    test_model_name = f"test-sdk-model-{str(uuid.uuid4())}"
    created_model = client.models.create_model(
        name=test_model_name,
        license_type="MIT",
        is_public=False,
        description="Test SDK model",
        description_short="Test SDK model",
        tasks=["OBJECT_DETECTION"],
        links=[],
        is_yolo=False,
    )

    # Assert model was created successfully
    assert created_model is not None
    assert hasattr(created_model, "id")
    assert created_model.name == test_model_name

    # Test deletion using the model ID (not the name)
    client.models.delete_model(identifier=created_model.id)
