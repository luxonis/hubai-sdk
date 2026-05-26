from __future__ import annotations

from collections.abc import Callable

import pytest
import requests

import hubai_sdk.services.instances as instance_services
import hubai_sdk.services.models as model_services
import hubai_sdk.services.variants as variant_services
from hubai_sdk.utils.hub import ResourceNotFoundError


def _model_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "name": "test-model",
        "license_type": "MIT",
        "is_public": False,
        "description_short": "short",
        "description": "desc",
        "architecture_id": None,
        "tasks": ["OBJECT_DETECTION"],
        "links": [],
        "is_yolo": False,
        "id": "aim_model",
        "team_name": None,
        "team_slug": None,
        "created": "2026-01-01T00:00:00.000000",
        "updated": "2026-01-01T00:00:00.000000",
        "slug": "test-model",
        "likes": 0,
        "downloads": 0,
        "versions": 1,
        "last_version_added": None,
        "platforms": [],
        "exportable_to": [],
        "project_id": None,
        "is_commercial": False,
        "is_official": False,
        "is_liked_by_user": False,
        "model_image_link": None,
        "exportable_types": [],
    }
    data.update(overrides)
    return data


def _variant_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "model_id": "aim_model",
        "name": "test-variant",
        "version": "0.1.0",
        "description": None,
        "repository_url": None,
        "commit_hash": None,
        "domain": None,
        "training_run_id": None,
        "training_run_name": None,
        "tags": [],
        "id": "aimv_variant",
        "created": "2026-01-01T00:00:00.000000",
        "updated": "2026-01-01T00:00:00.000000",
        "slug": "test-variant-0-1-0",
        "variant_slug": "test-variant",
        "platforms": [],
        "number_base_files": 0,
        "is_public": False,
        "exportable_to": [],
        "available_for": [],
        "exportable_types": [],
        "has_base_nn_archive": False,
    }
    data.update(overrides)
    return data


class _DummyConfig:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


def _assert_cli_exit_on_missing(
    monkeypatch: pytest.MonkeyPatch,
    action: Callable[[], object],
    expected_message: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("hubai_sdk.utils.hub.typer.echo", calls.append)

    with pytest.raises(Exception) as exc_info:
        action()

    assert exc_info.value.__class__.__name__ == "Exit"
    assert calls == [expected_message]


def test_list_models_returns_typed_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        model_services.Request, "get", lambda *args, **kwargs: [_model_data()]
    )

    models = model_services.list_models()

    assert len(models) == 1
    assert models[0].id == "aim_model"


def test_get_model_info_cli_prints_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(model_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        model_services,
        "get_resource_info",
        lambda identifier, endpoint: _model_data(),
    )
    monkeypatch.setattr(
        model_services,
        "print_hub_resource_info",
        lambda *args, **kwargs: calls.append(args),
    )

    model_services.get_model_info_cli("aim_model")

    assert calls


def test_get_model_info_cli_exits_cleanly_on_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_services,
        "get_model",
        lambda identifier: (_ for _ in ()).throw(
            ResourceNotFoundError(str(identifier), "models")
        ),
    )

    _assert_cli_exit_on_missing(
        monkeypatch,
        lambda: model_services.get_model_info_cli("missing-model"),
        "Resource for endpoint 'models' with identifier "
        "'missing-model' not found in HubAI.",
    )


def test_update_model_uses_resolved_resource_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(model_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        model_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "aim_model",
    )
    monkeypatch.setattr(
        model_services.Request,
        "patch",
        lambda *, service, endpoint, json: (
            calls.append((endpoint, json)),
            _model_data(id="aim_model"),
        )[1],
    )
    monkeypatch.setattr(
        model_services,
        "get_model",
        lambda identifier: model_services.ModelResponse(**_model_data()),
    )

    model_services.update_model("test-model", description="updated")

    assert calls == [("models/aim_model", {"description": "updated"})]


def test_get_variant_returns_typed_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(variant_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        variant_services,
        "get_resource_info",
        lambda identifier, endpoint: (
            _model_data() if endpoint == "models" else _variant_data()
        ),
    )

    variant = variant_services.get_variant("aimv_variant")

    assert variant.id == "aimv_variant"
    assert variant.model_name == "test-model"


def test_get_config_cli_logs_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(instance_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        instance_services, "ArchiveConfigurationResponse", _DummyConfig
    )
    monkeypatch.setattr(
        instance_services,
        "_get_instance_subresource",
        lambda identifier, subpath: {"config_version": "1.0"},
    )
    monkeypatch.setattr(
        instance_services.logger,
        "info",
        lambda *args, **kwargs: calls.append(args),
    )

    instance_services.get_config_cli("aimi_instance")

    assert calls


def test_download_instance_translates_not_found_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(instance_services, "get_telemetry", lambda: None)
    monkeypatch.setattr(
        instance_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "aimi_instance",
    )
    monkeypatch.setattr(
        instance_services.Request,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(404)),
    )

    with pytest.raises(ResourceNotFoundError):
        instance_services.download_instance("missing-instance")
