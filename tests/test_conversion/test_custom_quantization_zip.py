from pathlib import Path

import pytest

import hubai_sdk.services.instances as instances_service
from hubai_sdk.utils.quantization import normalize_quantization_input


@pytest.fixture
def quantization_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "quantization.zip"
    zip_path.write_bytes(b"zip-content")
    return zip_path


def test_upload_quantization_zip_uses_signed_put_url(
    quantization_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the SDK requests a signed URL and uploads the zip via
    PUT."""
    request_calls: list[tuple[str, str]] = []
    put_calls: list[dict[str, object]] = []

    monkeypatch.setattr(instances_service, "get_telemetry", lambda: None)

    def fake_post(
        service: str, endpoint: str, **kwargs: object
    ) -> dict[str, str]:
        request_calls.append((service, endpoint))
        return {"upload_url": "https://signed.example/upload"}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_put(
        url: str,
        data: bytes,
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        put_calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(instances_service.Request, "post", fake_post)
    monkeypatch.setattr(instances_service.requests, "put", fake_put)

    # This helper should not talk to the backend directly beyond requesting the
    # signed URL. The actual file upload goes to the returned storage URL.
    instances_service.upload_quantization_zip(str(quantization_zip), "job-123")

    assert request_calls == [
        (
            "models",
            "modelInstances/export/job-123/upload_quantization_zip",
        )
    ]
    assert len(put_calls) == 1
    assert put_calls[0]["url"] == "https://signed.example/upload"
    assert put_calls[0]["data"] == b"zip-content"
    assert put_calls[0]["headers"] == {"Content-Type": "application/zip"}


def test_convert_requires_zip_when_quantization_data_is_custom() -> None:
    with pytest.raises(ValueError, match="quantization_data='CUSTOM'"):
        normalize_quantization_input("CUSTOM")


def test_normalize_quantization_input_classifies_sources(
    quantization_zip: Path,
) -> None:
    assert normalize_quantization_input("GENERAL").input_type == (
        "predefined_dataset"
    )
    assert normalize_quantization_input("aid_dataset123").input_type == (
        "dataset_id"
    )
    normalized = normalize_quantization_input(str(quantization_zip))
    assert normalized.input_type == "custom_zip"
    assert normalized.quantization_data == "CUSTOM"
