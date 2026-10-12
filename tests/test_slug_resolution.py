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
            params = kwargs["params"]
            assert params == {
                "model_id": "aim_model",
                "variant_slug": "test-variant",
                "version": None,
                "is_public": params["is_public"],
                "limit": 500,
            }
            return [_variant_data()] if params["is_public"] else []
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
            params = kwargs["params"]
            assert params == {
                "model_id": "aim_model",
                "variant_slug": "test-variant",
                "version": "1.1.0",
                "is_public": params["is_public"],
                "limit": 500,
            }
            return (
                [
                    _variant_data(id="aimv_old", version="0.1.0"),
                    _variant_data(id="aimv_new", version="1.1.0"),
                ]
                if params["is_public"]
                else []
            )
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
            return (
                [
                    _variant_data(id="aimv_old", version="0.1.0"),
                    _variant_data(id="aimv_new", version="1.1.0"),
                ]
                if kwargs["params"]["is_public"]
                else []
            )
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
            params = kwargs["params"]
            assert params == {
                "model_version_id": "aimv_variant",
                "is_public": params["is_public"],
                "limit": 500,
            }
            return [_instance_data()] if params["is_public"] else []
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert (
        hub_utils.get_resource_id(identifier, "modelInstances")
        == "aimi_instance"
    )


def test_instance_hash_resolves_across_variant_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            return (
                [
                    _variant_data(id="aimv_old", version="0.1.0"),
                    _variant_data(id="aimv_new", version="1.0.0"),
                ]
                if kwargs["params"]["is_public"]
                else []
            )
        if endpoint == "modelInstances":
            params = kwargs["params"]
            if not params["is_public"]:
                return []
            if params["model_version_id"] == "aimv_old":
                return [_instance_data(id="aimi_old", hash_short="oldhash")]
            return [
                _instance_data(
                    id="aimi_new",
                    model_version_id="aimv_new",
                    hash_short="abc1234",
                )
            ]
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    assert (
        hub_utils.get_resource_id(
            "test-team/test-model:test-variant:abc1234", "modelInstances"
        )
        == "aimi_new"
    )


def test_instance_hash_shared_by_versions_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        if endpoint == "modelVersions":
            return (
                [
                    _variant_data(id="aimv_old", version="0.1.0"),
                    _variant_data(id="aimv_new", version="1.0.0"),
                ]
                if kwargs["params"]["is_public"]
                else []
            )
        if endpoint == "modelInstances":
            params = kwargs["params"]
            return (
                [
                    _instance_data(
                        id=f"aimi_{params['model_version_id']}",
                        model_version_id=params["model_version_id"],
                    )
                ]
                if params["is_public"]
                else []
            )
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(hub_utils.Request, "get", fake_get)

    with pytest.raises(ResourceAmbiguousError, match="modelInstances"):
        hub_utils.get_resource_id(
            "test-team/test-model:test-variant:abc1234", "modelInstances"
        )


def test_resolve_unique_variant_instance_by_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        instance_services,
        "resolve_variant_ids",
        lambda identifier: ["aimv_variant"],
    )

    def fake_get(*args: object, **kwargs: object) -> list[dict[str, object]]:
        params = kwargs["params"]
        assert params == {
            "model_version_id": "aimv_variant",
            "model_type": ModelType.ONNX,
            "is_public": params["is_public"],
            "limit": 500,
        }
        return (
            [{"id": "aimi_onnx", "model_type": "ONNX"}]
            if params["is_public"]
            else []
        )

    monkeypatch.setattr(instance_services.Request, "get", fake_get)

    assert (
        instance_services._resolve_variant_instance_id(
            "test-model:test-variant", ModelType.ONNX
        )
        == "aimi_onnx"
    )


def _mock_typed_variant_lookup(
    monkeypatch: pytest.MonkeyPatch,
    version_types: dict[str, ModelType | None],
) -> list[str]:
    queried_versions: list[str] = []

    def fake_get(
        service: str, endpoint: str, **kwargs: object
    ) -> list[dict[str, object]]:
        assert service == "models"
        if endpoint == "models/":
            return [_model_data()]
        params = kwargs["params"]
        if endpoint == "modelVersions":
            return (
                [
                    _variant_data(id=version_id, version=version)
                    for version_id, version in [
                        ("aimv_old", "1.0.0"),
                        ("aimv_new", "1.1.0"),
                    ]
                    if version_id in version_types
                ]
                if params["is_public"]
                else []
            )
        if endpoint == "modelInstances":
            version_id = params["model_version_id"]
            queried_versions.append(version_id)
            instance_type = version_types[version_id]
            return (
                [
                    _instance_data(
                        id=f"aimi_{version_id}",
                        model_version_id=version_id,
                        model_type=instance_type,
                    )
                ]
                if params["is_public"] and instance_type is not None
                else []
            )
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(instance_services.Request, "get", fake_get)
    return queried_versions


@pytest.mark.parametrize(
    ("identifier", "version_types", "expected", "queried_versions"),
    [
        (
            "test-team/test-model:test-variant",
            {"aimv_new": ModelType.ONNX},
            "aimi_aimv_new",
            {"aimv_new"},
        ),
        (
            "test-team/test-model:test-variant",
            {"aimv_old": ModelType.TFLITE, "aimv_new": ModelType.ONNX},
            "aimi_aimv_new",
            {"aimv_old", "aimv_new"},
        ),
        (
            "test-team/test-model:test-variant:1.1.0",
            {"aimv_old": ModelType.ONNX, "aimv_new": ModelType.ONNX},
            "aimi_aimv_new",
            {"aimv_new"},
        ),
    ],
)
def test_typed_instance_resolution_across_versions(
    monkeypatch: pytest.MonkeyPatch,
    identifier: str,
    version_types: dict[str, ModelType | None],
    expected: str,
    queried_versions: set[str],
) -> None:
    queried = _mock_typed_variant_lookup(monkeypatch, version_types)

    assert (
        instance_services._resolve_variant_instance_id(
            identifier, ModelType.ONNX
        )
        == expected
    )
    assert set(queried) == queried_versions


def test_typed_instance_resolution_is_ambiguous_across_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_typed_variant_lookup(
        monkeypatch,
        {"aimv_old": ModelType.ONNX, "aimv_new": ModelType.ONNX},
    )

    with pytest.raises(ResourceAmbiguousError, match="modelInstances"):
        instance_services._resolve_variant_instance_id(
            "test-team/test-model:test-variant", ModelType.ONNX
        )
