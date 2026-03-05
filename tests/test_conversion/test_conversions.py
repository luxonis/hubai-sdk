import os
import shutil
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

from hubai_sdk import HubAIClient
from hubai_sdk.utils.sdk_models import ConvertResponse

os.environ["HUBAI_TELEMETRY_ENABLED"] = "false"


def _cleanup_response(
    response: ConvertResponse | None, client: HubAIClient
) -> None:
    if response is None:
        return

    downloaded_path = response.downloaded_path.resolve()
    with suppress(FileNotFoundError):
        shutil.rmtree(str(downloaded_path.parent))

    with suppress(Exception):
        client.models.delete_model(str(response.instance.model_id))


def _assert_response_downloaded(response: ConvertResponse) -> None:
    downloaded_path = response.downloaded_path.resolve()
    assert Path(downloaded_path).exists()


def test_rvc2_conversion(client: HubAIClient, base_model_path: str):
    model_name = f"test-sdk-conversion-rvc2-{uuid.uuid4()}"
    response: ConvertResponse | None = None

    try:
        response = client.convert.RVC2(
            path=base_model_path,
            name=model_name,
        )
        _assert_response_downloaded(response)
    finally:
        _cleanup_response(response, client)


def test_rvc2_legacy_conversion(client: HubAIClient, base_model_path: str):
    model_name = f"test-sdk-conversion-rvc2-legacy-{uuid.uuid4()}"
    response: ConvertResponse | None = None

    try:
        response = client.convert.RVC2(
            path=base_model_path,
            name=model_name,
            superblob=False,
        )
        _assert_response_downloaded(response)
    finally:
        _cleanup_response(response, client)


@pytest.mark.parametrize(
    ("quantization_mode", "requires_calibration"),
    [
        ("INT8_STANDARD", True),
        ("INT8_ACCURACY_FOCUSED", True),
        ("INT8_INT16_MIXED", True),
        ("INT8_INT16_MIXED_ACCURACY_FOCUSED", True),
        ("FP16_STANDARD", False),
    ],
)
def test_rvc4_conversion_all_quantization_modes(
    client: HubAIClient,
    base_model_path: str,
    quantization_mode: str,
    requires_calibration: bool,
):
    model_name = (
        f"test-sdk-conversion-rvc4-{quantization_mode.lower()}-{uuid.uuid4()}"
    )

    convert_kwargs = {
        "path": base_model_path,
        "name": model_name,
        "quantization_mode": quantization_mode,
    }

    if requires_calibration:
        convert_kwargs["quantization_data"] = "GENERAL"
        convert_kwargs["max_quantization_images"] = 50

    response: ConvertResponse | None = None

    try:
        response = client.convert.RVC4(**convert_kwargs)
        _assert_response_downloaded(response)
    finally:
        _cleanup_response(response, client)
