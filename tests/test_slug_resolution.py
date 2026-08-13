from __future__ import annotations

import pytest

import hubai_sdk.services.instances as instance_services
import hubai_sdk.utils.hub as hub_utils
from hubai_sdk.errors import ResourceAmbiguousError
from hubai_sdk.utils.types import ModelType


def _model_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "aim_model",
        "slug": "test-model",
        "team_slug": "test-team",
    }
    data.update(overrides)
    return data


def _variant_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "aimv_variant",
        "model_id": "aim_model",
        "slug": "test-variant-1.0.0",
        "variant_slug": "test-variant",
        "version": "1.0.0",
    }
    data.update(overrides)
    return data


def _instance_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "aimi_instance",
        "model_version_id": "aimv_variant",
        "slug": "test-instance",
        "hash_short": "abc1234",
    }
    data.update(overrides)
    return data


def test_resolve_team_model_resource_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        assert endpoint == "models/"
        return [_model_data()]

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert (
        hub_utils.get_resource_id("test-team/test-model", "models")
        == "aim_model"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "test-model:test-variant",
        "test-team/test-model:test-variant",
    ],
)
def test_resolve_variant_resource_path(
    monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            assert kwargs["params"] == {
                "model_id": "aim_model",
                "variant_slug": "test-variant",
                "version": None,
                "limit": 500,
            }
            return [_variant_data()]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert (
        hub_utils.get_resource_id(identifier, "modelVersions")
        == "aimv_variant"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "test-model:test-variant:1.1.0",
        "test-team/test-model:test-variant:1.1.0",
    ],
)
def test_resolve_versioned_variant_resource_path(
    monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            assert kwargs["params"] == {
                "model_id": "aim_model",
                "variant_slug": "test-variant",
                "version": "1.1.0",
                "limit": 500,
            }
            return [
                _variant_data(id="aimv_old", version="0.1.0"),
                _variant_data(id="aimv_new", version="1.1.0"),
            ]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert hub_utils.get_resource_id(identifier, "modelVersions") == "aimv_new"


def test_unversioned_variant_resource_path_is_ambiguous_across_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            return [
                _variant_data(id="aimv_old", version="0.1.0"),
                _variant_data(id="aimv_new", version="1.1.0"),
            ]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    with pytest.raises(ResourceAmbiguousError, match="ambiguous"):
        hub_utils.get_resource_id(
            "test-team/test-model:test-variant", "modelVersions"
        )


def test_ambiguous_simple_slug_raises_a_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_utils.Request,
        "get",
        lambda *args, **kwargs: [
            _model_data(id="aim_model_one"),
            _model_data(id="aim_model_two"),
        ],
    )

    with pytest.raises(ResourceAmbiguousError, match="ambiguous"):
        hub_utils.get_resource_id("test-model", "models")


@pytest.mark.parametrize(
    "identifier",
    [
        "test-model:test-variant:abc1234",
        "test-team/test-model:test-variant:abc1234",
    ],
)
def test_resolve_instance_resource_path(
    monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            return [_variant_data()]
        if endpoint == "modelInstances":
            assert kwargs["params"] == {
                "model_version_id": "aimv_variant",
                "limit": 500,
            }
            return [_instance_data()]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert (
        hub_utils.get_resource_id(identifier, "modelInstances")
        == "aimi_instance"
    )


def test_resolve_unique_variant_instance_by_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        instance_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "aimv_variant",
    )
    monkeypatch.setattr(
        instance_services.Request,
        "get",
        lambda *args, **kwargs: [{"id": "aimi_onnx", "model_type": "ONNX"}],
    )

    assert (
        instance_services._resolve_variant_instance_id(
            "test-model:test-variant", ModelType.ONNX
        )
        == "aimi_onnx"
    )
