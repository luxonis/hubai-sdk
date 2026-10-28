from pathlib import Path

import pytest
import requests

import hubai_sdk.services.architectures as architecture_services
from hubai_sdk import HubAIClient
from hubai_sdk.errors import (
    InputError,
    ResourceAmbiguousError,
    ResourceNotFoundError,
)
from hubai_sdk.utils import hub as hub_utils
from hubai_sdk.utils import hub_requests

TESTS_DIR = Path(__file__).parent


def _architecture_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "name": "test-architecture",
        "description": "desc",
        "license_type": "MIT",
        "is_public": False,
        "tasks": ["CLASSIFICATION"],
        "training_image": "luxonis-train",
        "training_model_config": {"name": "resnet18"},
        "training_loader_config": {"name": "LuxonisLoaderTorch"},
        "training_hyperparameter_config": {"epochs": 1},
        "performance_ratings": {
            "rvc2_ratings": {
                "accuracy_rating": None,
                "speed_rating": None,
                "fps_rating": None,
            },
            "rvc3_ratings": {
                "accuracy_rating": None,
                "speed_rating": None,
                "fps_rating": None,
            },
            "rvc4_ratings": {
                "accuracy_rating": None,
                "speed_rating": None,
                "fps_rating": None,
            },
            "hailo_ratings": {
                "accuracy_rating": None,
                "speed_rating": None,
                "fps_rating": None,
            },
        },
        "id": "aia_architecture",
        "created": "2026-01-01T00:00:00.000000",
        "updated": "2026-01-01T00:00:00.000000",
        "slug": "test-architecture",
        "is_owner": True,
    }
    data.update(overrides)
    return data


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


def test_list_architectures_returns_typed_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        architecture_services.Request,
        "get",
        lambda *args, **kwargs: [_architecture_data()],
    )

    architectures = architecture_services.list_architectures()

    assert len(architectures) == 1
    assert architectures[0].id == "aia_architecture"
    assert architectures[0].name == "test-architecture"
    assert "team_id" not in type(architectures[0]).model_fields
    assert "user_id" not in type(architectures[0]).model_fields


def test_get_architecture_info_cli_prints_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        architecture_services,
        "get_resource_info",
        lambda identifier, endpoint: _architecture_data(),
    )
    monkeypatch.setattr(
        architecture_services,
        "print_hub_resource_info",
        lambda *args, **kwargs: calls.append(args),
    )

    architecture_services.get_architecture_info_cli("aia_architecture")

    assert calls


def test_get_architecture_info_cli_exits_cleanly_on_missing_architecture(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: (_ for _ in ()).throw(
            ResourceNotFoundError(str(identifier), "architectures")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        architecture_services.get_architecture_info_cli("missing-architecture")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Resource for endpoint 'architectures' with identifier "
        "'missing-architecture' not found in HubAI.\n"
    )


def test_create_architecture_requires_training_config() -> None:
    with pytest.raises(InputError) as exc_info:
        architecture_services.create_architecture("missing-config")

    assert str(exc_info.value) == (
        "Must provide either config_file_path or both "
        "training_model_config and training_hyperparameter_config"
    )


def test_create_architecture_rejects_non_mapping_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("null\n")
    with pytest.raises(InputError, match="must contain a mapping"):
        architecture_services.create_architecture(
            "invalid-config", config_file_path=str(config_path)
        )


def test_create_architecture_rejects_null_visibility() -> None:
    with pytest.raises(InputError):
        architecture_services.create_architecture(
            "invalid-visibility",
            is_public=None,  # type: ignore[arg-type]
            training_model_config={"name": "resnet18"},
            training_hyperparameter_config={"epochs": 1},
        )


def test_update_architecture_uses_resolved_resource_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )
    monkeypatch.setattr(
        architecture_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "aia_architecture",
    )
    monkeypatch.setattr(
        architecture_services.Request,
        "patch",
        lambda *, service, endpoint, json: (
            calls.append((endpoint, json)),
            _architecture_data(description=json["description"]),
        )[1],
    )

    updated = architecture_services.update_architecture(
        "test-architecture", description="updated"
    )

    assert updated.description == "updated"
    assert calls[0][0] == "architectures/aia_architecture"


def test_delete_architecture_translates_not_found_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        architecture_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "aia_missing",
    )
    monkeypatch.setattr(
        architecture_services.Request,
        "delete",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(404)),
    )

    with pytest.raises(ResourceNotFoundError):
        architecture_services.delete_architecture("missing-architecture")


def test_architecture_slug_lookup_follows_pages_and_matches_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_get_page(
        *, service: str, endpoint: str, params: dict[str, object]
    ) -> tuple[list[dict[str, object]], str | None]:
        assert service == "models"
        assert endpoint == "architectures/"
        calls.append(params)
        if params["cursor"] is None:
            return (
                [{"id": "aia_other", "slug": "test-architecture-other"}],
                "next",
            )
        return (
            [{"id": "aia_architecture", "slug": "test-architecture"}],
            None,
        )

    monkeypatch.setattr(hub_utils.Request, "get_page", fake_get_page)
    monkeypatch.setattr(
        hub_utils.Request,
        "get",
        lambda **kwargs: _architecture_data(),
    )

    result = architecture_services.get_architecture("test-architecture")

    assert result.id == "aia_architecture"
    assert len(calls) == 4
    assert [call["cursor"] for call in calls] == [None, "next", None, "next"]
    assert all("search" not in call and "slug" not in call for call in calls)


def test_get_page_returns_cursor_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b'[{"id": "aia_architecture"}]'
    response.headers["X-Next-Cursor"] = "opaque-cursor"
    monkeypatch.setattr(
        hub_requests.Request,
        "headers",
        lambda: {"Authorization": "Bearer local-test-key"},
    )
    monkeypatch.setattr(
        hub_requests.requests,
        "get",
        lambda *args, **kwargs: response,
    )

    data, cursor = hub_requests.Request.get_page(
        "models", "architectures", params={"limit": 1}
    )

    assert data == [{"id": "aia_architecture"}]
    assert cursor == "opaque-cursor"


def test_architecture_slug_lookup_rejects_ambiguous_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_utils.Request,
        "get_page",
        lambda **kwargs: (
            [
                {"id": "aia_one", "slug": "duplicate"},
                {"id": "aia_two", "slug": "duplicate"},
            ],
            None,
        ),
    )
    with pytest.raises(ResourceAmbiguousError):
        architecture_services.get_architecture("duplicate")


def test_architecture_slug_lookup_rejects_fuzzy_only_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_utils.Request,
        "get_page",
        lambda **kwargs: ([{"id": "aia_other", "slug": "other"}], None),
    )
    with pytest.raises(ResourceNotFoundError):
        architecture_services.get_architecture("missing")


def test_list_architectures_filters_license_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_get_page(
        *, service: str, endpoint: str, params: dict[str, object]
    ) -> tuple[list[dict[str, object]], str | None]:
        calls.append(params)
        if params["cursor"] is None:
            return (
                [
                    _architecture_data(
                        id="aia_other", license_type="Apache 2.0"
                    )
                ],
                "next",
            )
        return ([_architecture_data()], None)

    monkeypatch.setattr(
        architecture_services.Request, "get_page", fake_get_page
    )
    result = architecture_services.list_architectures(
        license_type="MIT", limit=1
    )
    assert [item.id for item in result] == ["aia_architecture"]
    assert "license_type" not in calls[0]
    assert calls[0]["limit"] == 500
    assert [call["cursor"] for call in calls] == [None, "next"]


def test_create_architecture_posts_validated_inline_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        architecture_services.Request,
        "post",
        lambda *, service, endpoint, json: (
            calls.append(json),
            {"id": "aia_architecture", "name": "test-architecture"},
        )[1],
    )
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )

    result = architecture_services.create_architecture(
        "test-architecture",
        training_model_config={"name": "resnet18"},
        training_hyperparameter_config={"epochs": 1},
    )

    assert result.id == "aia_architecture"
    assert calls[0]["description"] == ""
    assert calls[0]["tasks"] == []
    assert "config_file_path" not in calls[0]


def test_update_architecture_sends_only_changed_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )
    monkeypatch.setattr(
        architecture_services.Request,
        "patch",
        lambda *, service, endpoint, json: (
            calls.append({"endpoint": endpoint, "json": json}),
            _architecture_data(description="updated"),
        )[1],
    )

    result = architecture_services.update_architecture(
        "test-architecture", description="updated"
    )

    assert result.description == "updated"
    assert calls == [
        {
            "endpoint": "architectures/aia_architecture",
            "json": {"description": "updated"},
        }
    ]


def test_update_architecture_replaces_training_config_from_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )
    monkeypatch.setattr(
        architecture_services.Request,
        "patch",
        lambda *, service, endpoint, json: (
            calls.append(json),
            _architecture_data(),
        )[1],
    )

    architecture_services.update_architecture(
        "aia_architecture",
        config_file_path=str(TESTS_DIR / "architecture.yaml"),
    )

    assert set(calls[0]) == {
        "training_model_config",
        "training_loader_config",
        "training_hyperparameter_config",
    }
    assert calls[0]["training_loader_config"]["name"] == "LuxonisLoaderTorch"


def test_update_architecture_rejects_file_and_inline_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )
    with pytest.raises(InputError, match="Cannot provide both"):
        architecture_services.update_architecture(
            "aia_architecture",
            config_file_path=str(TESTS_DIR / "architecture.yaml"),
            training_model_config={"name": "other"},
        )


def test_update_architecture_without_changes_does_not_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        architecture_services,
        "get_architecture",
        lambda identifier: architecture_services.ArchitectureResponse(
            **_architecture_data()
        ),
    )
    monkeypatch.setattr(
        architecture_services.Request,
        "patch",
        lambda **kwargs: pytest.fail("unexpected PATCH"),
    )
    assert (
        architecture_services.update_architecture("aia_architecture").id
        == "aia_architecture"
    )


def test_client_exposes_builtin_architecture_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HubAIClient, "_verify_api_key", lambda self: True)
    client = HubAIClient(api_key="local-test-key")
    assert client.architecture is architecture_services
