"""Pytest configuration and fixtures for the HubAI SDK tests."""

import os
from pathlib import Path

import pytest

from hubai_sdk import HubAIClient


def pytest_addoption(parser: pytest.Parser):
    """Add command line options for tests."""
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to the model file for testing",
    )

    parser.addoption(
        "--yolo-version",
        action="store",
        default=None,
        help="YOLO version to use for testing (e.g., yolov8, yolov5)",
    )

    parser.addoption(
        "--base-instance-id",
        action="store",
        default=None,
        help="Base instance ID to download for conversion tests",
    )


@pytest.fixture
def client():
    """Fixture to provide HubAI client instance."""
    api_key = os.getenv("HUBAI_API_KEY")
    if not api_key:
        pytest.skip("HUBAI_API_KEY environment variable not set")
    return HubAIClient(api_key=api_key)


@pytest.fixture
def model_path(request: pytest.FixtureRequest):
    """Fixture to provide model path from command line."""
    path = request.config.getoption("--model-path")

    # Check if file exists
    if not Path.exists(path):
        pytest.skip(f"Model file not found at: {path}")

    return path


@pytest.fixture
def yolo_version(request: pytest.FixtureRequest):
    """Fixture to provide YOLO version from command line."""
    return request.config.getoption("--yolo-version")


@pytest.fixture
def base_model_path(request: pytest.FixtureRequest, client: HubAIClient):
    """Fixture to provide base model path for conversion tests.

    Downloads the base instance if not already cached locally. The
    instance ID can be provided via --base-instance-id command line
    option or HUBAI_BASE_INSTANCE_ID environment variable. Defaults to a
    hardcoded instance ID if neither is provided.
    """
    # Try to get instance ID from command line, env var, or use default
    instance_id = request.config.getoption("--base-instance-id", default=None)
    if instance_id is None:
        instance_id = os.getenv("HUBAI_BASE_INSTANCE_ID", "57fb6e9b-3157-44e7-9a1d-40254deb5313")
        if os.getenv("HUBAI_STAGE", "") == "stg":
            instance_id = "aimi_WALJ4SHoZWaaPoXDEWmEmP_stg"

    base_instance = client.instances.get_instance(instance_id)
    if base_instance is None:
        raise ValueError(f"Base instance with ID {instance_id} not found")

    downloaded_path = client.instances.download_instance(base_instance.id, output_dir=str(Path.cwd()))
    return downloaded_path


@pytest.fixture
def test_instance_id():
    """Fixture to provide a test instance ID for instance operations."""
    stg_instance_id = "aimi_WALJ4SHoZWaaPoXDEWmEmP_stg"
    prod_instance_id = "57fb6e9b-3157-44e7-9a1d-40254deb5313"
    return stg_instance_id if os.getenv("HUBAI_STAGE", "") == "stg" else prod_instance_id


@pytest.fixture
def test_model_id():
    """Fixture to provide a test model ID for variant operations."""
    stg_model_id = "aim_UJuH8qn9Q2XDoNw36Ljy2Z_stg"
    prod_model_id = "252c6e74-2869-4cbb-af7b-ccc9d655b42f"
    return stg_model_id if os.getenv("HUBAI_STAGE", "") == "stg" else prod_model_id
