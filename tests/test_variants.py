import pytest
from loguru import logger
import os
from hubai_sdk import HubAIClient
import uuid
os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"

@pytest.fixture
def test_model_id():
    """Fixture to provide a test model ID for variant operations."""
    return "252c6e74-2869-4cbb-af7b-ccc9d655b42f"


def test_list_variants(client: HubAIClient, test_model_id: str):
    """Test listing variants for a specific model."""
    variants = client.variants.list_variants(model_id=test_model_id)
    assert variants is not None

    assert isinstance(variants, list)
    assert len(variants) >= 0


def test_get_variant(client: HubAIClient, test_model_id: str):
    """Test getting a specific variant."""
    variants = client.variants.list_variants(model_id=test_model_id)
    if not variants:
        pytest.skip("No variants available to test get_variant")

    selected_variant = variants[0]
    variant = client.variants.get_variant(selected_variant.id)

    assert variant is not None
    assert hasattr(variant, "id")
    assert variant.id == selected_variant.id


def test_create_and_delete_variant(client: HubAIClient, test_model_id: str):
    """Test creating and deleting a variant."""
    variant_name = f"test-sdk-variant-{str(uuid.uuid4())}"
    created_variant = client.variants.create_variant(
        name=variant_name,
        model_id=test_model_id,
        variant_version="1.0.0",
    )

    # Assert variant was created successfully
    assert created_variant is not None
    assert hasattr(created_variant, "id")
    assert created_variant.name == variant_name
    assert created_variant.version == "1.0.0"
    assert created_variant.model_id == test_model_id

    # Test deletion using the variant ID
    variant_id = created_variant.id
    client.variants.delete_variant(variant_id)
