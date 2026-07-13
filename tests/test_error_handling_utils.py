from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import hubai_sdk.services.convert as convert_service
import hubai_sdk.utils.hub as hub_utils
from hubai_sdk.errors import (
    HubApiError,
    ValidationError,
)
from hubai_sdk.hubai_client import HubAIClient


def _http_error(
    status_code: int, payload: object | None = None
) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    if payload is not None:
        response._content = json.dumps(payload).encode()
        response.headers["Content-Type"] = "application/json"
    return requests.HTTPError(response=response)


def test_resolve_resource_id_does_not_mask_lookup_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_utils.Request,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(401)),
    )

    with pytest.raises(HubApiError, match="401"):
        hub_utils.resolve_resource_id("missing-model", "models")


def test_full_slug_to_id_falls_back_on_invalid_slug_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_utils.Request,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            _http_error(
                422,
                {
                    "detail": [
                        {
                            "loc": ["body"],
                            "msg": (
                                "Value error, Invalid slug format. Slugs "
                                "must be in the format {team-slug}/"
                                "{model-slug}:{variant-slug}."
                            ),
                        }
                    ]
                },
            )
        ),
    )

    assert hub_utils.full_slug_to_id("plain-slug", "models") is None


def test_convert_does_not_swallow_validation_error_from_create_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = SimpleNamespace(
        input_model=Path("model.onnx"),
        inputs=[SimpleNamespace(shape=[1, 3, 640, 640], layout="NCHW")],
    )
    config = SimpleNamespace(name="demo-model", stages={"main": stage})

    monkeypatch.setattr(convert_service, "is_nn_archive", lambda path: False)
    monkeypatch.setattr(
        convert_service,
        "get_configs",
        lambda path, opts: (config, None, "main"),
    )
    monkeypatch.setattr(
        convert_service, "cleanup_extracted_path", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        convert_service,
        "create_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValidationError("bad create", status_code=422)
        ),
    )
    monkeypatch.setattr(
        convert_service,
        "resolve_resource_id",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resolve_resource_id should not be called")
        ),
    )

    with pytest.raises(ValidationError, match="bad create"):
        convert_service.convert(convert_service.Target.RVC4, path="model.yaml")


def test_hubai_client_verify_api_key_handles_auth_and_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HubAIClient.__new__(HubAIClient)
    monkeypatch.setattr(
        "hubai_sdk.hubai_client.Request.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(401)),
    )

    assert client._verify_api_key() is False

    monkeypatch.setattr(
        "hubai_sdk.hubai_client.Request.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            _http_error(500, {"detail": "backend down"})
        ),
    )

    with pytest.raises(HubApiError, match="backend down"):
        client._verify_api_key()
