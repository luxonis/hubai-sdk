from __future__ import annotations

import pytest

import hubai_sdk.services.convert as convert_service
from hubai_sdk.utils.hub import ResourceNotFoundError
from hubai_sdk.utils.hubai_models import JobMessageResponse


def _instance_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "model_version_id": "aimv_variant",
        "parent_id": None,
        "model_type": "RVC4",
        "name": "exported-instance",
        "description": None,
        "tags": [],
        "job_id": None,
        "hardware_parameters": None,
        "input_shape": None,
        "quantization_data": None,
        "yolo_version": None,
        "id": "aimi_exported",
        "created": "2026-01-01T00:00:00.000000",
        "updated": "2026-01-01T00:00:00.000000",
        "slug": "exported-instance",
        "is_nn_archive": False,
        "model_class": None,
        "model_id": "aim_model",
        "exportable_to": [],
        "is_public": False,
        "model_precision_type": "FP16",
        "status": "available",
        "platforms": ["RVC4"],
        "hash": None,
        "hash_short": None,
        "model_version_name": None,
        "training_run_name": None,
    }
    data.update(overrides)
    return data


def test_resolve_exported_instance_fetches_raw_instance_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_calls: list[tuple[str, str]] = []

    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> dict[str, object] | list[str]:
        request_calls.append((service, endpoint))
        if endpoint == "modelInstances/aimi_exported/":
            return _instance_data()
        if endpoint == "modelInstances/aimi_exported/download":
            return ["https://example.invalid/model.blob"]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(convert_service.Request, "get", fake_get)
    monkeypatch.setattr(convert_service, "sleep", lambda _: None)

    job = JobMessageResponse(
        id="aij_export",
        arguments={},
        extra={},
        name="export job",
        status="COMPLETED",
        result={"resulting_model_instance_id": "aimi_exported"},
    )

    instance = convert_service._resolve_exported_instance(job)

    assert instance.id == "aimi_exported"
    assert request_calls == [
        ("models", "modelInstances/aimi_exported/"),
        ("models", "modelInstances/aimi_exported/download"),
    ]


def test_resolve_exported_instance_retries_until_instance_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_calls: list[str] = []
    attempts = {"instance_reads": 0}

    def fake_get_resource_info(
        identifier: str, endpoint: str
    ) -> dict[str, object]:
        request_calls.append(f"{endpoint}:{identifier}")
        if endpoint == "modelInstances":
            attempts["instance_reads"] += 1
            if attempts["instance_reads"] == 1:
                raise ResourceNotFoundError(identifier, endpoint)
            return _instance_data()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    def fake_get(service: str, endpoint: str, **kwargs: object) -> list[str]:
        if endpoint == "modelInstances/aimi_exported/download":
            return ["https://example.invalid/model.blob"]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(
        convert_service, "get_resource_info", fake_get_resource_info
    )
    monkeypatch.setattr(convert_service.Request, "get", fake_get)
    monkeypatch.setattr(convert_service, "sleep", lambda _: None)

    job = JobMessageResponse(
        id="aij_export",
        arguments={},
        extra={},
        name="export job",
        status="COMPLETED",
        result={"resulting_model_instance_id": "aimi_exported"},
    )

    instance = convert_service._resolve_exported_instance(job)

    assert instance.id == "aimi_exported"
    assert request_calls == [
        "modelInstances:aimi_exported",
        "modelInstances:aimi_exported",
    ]
