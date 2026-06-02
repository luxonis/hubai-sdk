"""Pytest configuration and fixtures for the HubAI SDK tests."""

import os
from pathlib import Path

import pytest

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"

from hubai_sdk import HubAIClient
from hubai_sdk.utils.types import ModelType


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

    parser.addoption(
        "--pytorch-variant-id",
        action="store",
        default=None,
        help="PyTorch model variant ID to use for conversion tests",
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
        instance_id = os.getenv(
            "HUBAI_BASE_INSTANCE_ID", "aimi_LCEFX2rJSsMhEjsyeMEcWn"
        )
        if os.getenv("HUBAI_STAGE", "") == "stg":
            instance_id = "aimi_WALJ4SHoZWaaPoXDEWmEmP_stg"

    base_instance = client.instances.get_instance(instance_id)
    if base_instance is None:
        raise ValueError(f"Base instance with ID {instance_id} not found")

    return client.instances.download_instance(
        base_instance.id, output_dir=str(Path.cwd())
    )


@pytest.fixture
def pytorch_variant_id(request: pytest.FixtureRequest):
    """Fixture to provide a stage-specific PyTorch variant ID."""
    variant_id = request.config.getoption("--pytorch-variant-id", default=None)
    if variant_id is not None:
        return variant_id

    variant_id = os.getenv("HUBAI_PYTORCH_VARIANT_ID")
    if variant_id is not None:
        return variant_id

    stg_variant_id = "aimv_7yxPWY65q2dSCqK84JsAFL_stg"
    prod_variant_id = "aimv_8UbzZDbeDpJzyARRnia9Yv"
    return (
        stg_variant_id
        if os.getenv("HUBAI_STAGE", "") == "stg"
        else prod_variant_id
    )


@pytest.fixture
def pytorch_model_path(
    client: HubAIClient,
    pytorch_variant_id: str,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Fixture to download a PyTorch source model for conversion
    tests."""
    instances = client.instances.list_instances(
        variant_id=pytorch_variant_id,
        model_type=ModelType.PYTORCH,
        model_class="base",
        status="available",
        limit=10,
    )

    if not instances:
        raise ValueError(
            "No available PyTorch base instance found for variant "
            f"{pytorch_variant_id}."
        )

    download_dir = tmp_path_factory.mktemp("pytorch-model")
    downloaded_path = Path(
        client.instances.download_instance(
            instances[0].id, output_dir=str(download_dir)
        )
    )

    if downloaded_path.suffix in {".pt", ".pth"}:
        return downloaded_path

    for pattern in ("*.pt", "*.pth"):
        matches = sorted(download_dir.glob(pattern))
        if matches:
            return matches[0]

    raise ValueError(
        "Downloaded PyTorch base instance does not contain a .pt or .pth file."
    )


@pytest.fixture
def test_instance_id():
    """Fixture to provide a test instance ID for instance operations."""
    stg_instance_id = "aimi_WALJ4SHoZWaaPoXDEWmEmP_stg"
    prod_instance_id = "aimi_LCEFX2rJSsMhEjsyeMEcWn"
    return (
        stg_instance_id
        if os.getenv("HUBAI_STAGE", "") == "stg"
        else prod_instance_id
    )


@pytest.fixture
def test_model_id():
    """Fixture to provide a test model ID for variant operations."""
    stg_model_id = "aim_UJuH8qn9Q2XDoNw36Ljy2Z_stg"
    prod_model_id = "aim_FNtJ9PtPdS54T833mCBRGi"
    return (
        stg_model_id
        if os.getenv("HUBAI_STAGE", "") == "stg"
        else prod_model_id
    )


@pytest.fixture
def test_variant_id():
    """Fixture to provide a test variant ID for instance operations."""
    stg_variant_id = "aimv_UJuH8pVjm1mvWGYd69Jioz_stg"
    prod_variant_id = "aimv_VitM2h2uQHtnZfQRJ2Ann9"
    return (
        stg_variant_id
        if os.getenv("HUBAI_STAGE", "") == "stg"
        else prod_variant_id
    )
