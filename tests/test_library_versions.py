"""Library-version service tests; all HTTP responses are local fixtures."""

import json

import pytest
import requests

from hubai_sdk import HubAIClient
from hubai_sdk.errors import InputError
from hubai_sdk.services import library_versions
from hubai_sdk.utils.environ import environ


def _response(
    payload: dict[str, object], status: int = 200
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://easyml.cloud.luxonis.com/",
            "https://easyml.cloud.luxonis.com/projects/api/v1/health/",
        ),
        (
            "https://easyml.cloud-stg.luxonis.com/",
            "https://easyml.cloud-stg.luxonis.com/projects/api/v1/health/",
        ),
        (
            "https://models.stg.hubai/",
            "https://models.stg.hubai/api/v1/health/",
        ),
    ],
)
def test_library_versions_url(
    monkeypatch: pytest.MonkeyPatch, base_url: str, expected: str
) -> None:
    monkeypatch.setattr(environ, "HUBAI_URL", base_url)
    assert library_versions._get_library_versions_url() == expected


def test_get_library_versions_extracts_only_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_get(url: str, *, timeout: int) -> requests.Response:
        calls.append((url, timeout))
        return _response(
            {
                "version": "server-commit",
                "luxonis_ml_version": "0.8.6",
                "luxonis_train_version": "0.4.6",
                "other": "ignored",
            }
        )

    monkeypatch.setattr(
        environ, "HUBAI_URL", "https://easyml.cloud.luxonis.com/"
    )
    monkeypatch.setattr(library_versions.requests, "get", fake_get)

    assert library_versions.get_library_versions() == {
        "luxonis-ml": "0.8.6",
        "luxonis-train": "0.4.6",
    }
    assert calls == [
        ("https://easyml.cloud.luxonis.com/projects/api/v1/health/", 200)
    ]


def test_no_library_versions_is_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        library_versions.requests,
        "get",
        lambda *args, **kwargs: _response({"version": "server-commit"}),
    )
    with pytest.raises(InputError, match="No library versions found"):
        library_versions.get_library_versions()


def test_client_exposes_builtin_library_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HubAIClient, "_verify_api_key", lambda self: True)
    client = HubAIClient(api_key="local-test-key")
    assert client.library_versions is library_versions
