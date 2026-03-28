import json
import shutil
import tarfile
import uuid
import zipfile
from contextlib import suppress
from io import BytesIO
from pathlib import Path

import numpy as np
import onnx
import pytest

from hubai_sdk import HubAIClient
from hubai_sdk.utils.sdk_models import ConvertResponse


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


def test_rvc4_conversion_with_custom_quantization_zip(
    client: HubAIClient,
    base_model_path: str,
    tmp_path: Path,
):
    model_name = f"test-sdk-conversion-rvc4-custom-quant-{uuid.uuid4()}"
    response: ConvertResponse | None = None

    input_shape = _get_input_shape_from_model(Path(base_model_path))
    quantization_zip = _create_quantization_zip(
        tmp_path / "quantization.zip", input_shape
    )

    try:
        response = client.convert.RVC4(
            path=base_model_path,
            name=model_name,
            quantization_mode="INT8_STANDARD",
            quantization_data=str(quantization_zip),
            max_quantization_images=2,
        )
        _assert_response_downloaded(response)
    finally:
        _cleanup_response(response, client)


def _create_quantization_zip(
    zip_path: Path, input_shape: list[int], num_samples: int = 2
) -> Path:
    sample_shape = (
        input_shape[1:] if input_shape and input_shape[0] == 1 else input_shape
    )
    if not sample_shape:
        sample_shape = [3, 288, 512]

    with zipfile.ZipFile(zip_path, "w") as archive:
        for index in range(num_samples):
            buffer = BytesIO()
            np.save(
                buffer,
                np.zeros(sample_shape, dtype=np.float32),
                allow_pickle=False,
            )
            archive.writestr(f"sample_{index}.npy", buffer.getvalue())

    return zip_path


def _get_input_shape_from_model(model_path: Path) -> list[int]:
    if tarfile.is_tarfile(model_path):
        with tarfile.open(model_path) as archive:
            config_member = archive.extractfile("config.json")
            if config_member is None:
                raise ValueError(
                    f"NN archive '{model_path}' does not contain config.json."
                )
            config = json.load(config_member)
        return list(config["model"]["inputs"][0]["shape"])

    model = onnx.load(model_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]
    if any(not dim for dim in input_shape):
        raise ValueError(
            f"Unable to determine a static input shape from '{model_path}'."
        )
    return input_shape
