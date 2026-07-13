from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import hubai_sdk.services.models as model_services
from hubai_sdk.hubai_client import HubAIClient

from .test_cli_resource_return_values import _model_data


@dataclass
class _FakeTelemetry:
    events: list[tuple[str, dict[str, object], dict[str, object]]] = field(
        default_factory=list
    )

    def capture(
        self,
        event: str,
        properties: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self.events.append((event, properties or {}, dict(kwargs)))


@pytest.fixture
def fake_telemetry(monkeypatch: pytest.MonkeyPatch) -> _FakeTelemetry:
    telemetry = _FakeTelemetry()
    monkeypatch.setattr(
        "hubai_sdk.utils.telemetry.get_component_telemetry",
        lambda: telemetry,
    )
    return telemetry


def test_list_models_emits_summary_and_operation_result(
    monkeypatch: pytest.MonkeyPatch, fake_telemetry: _FakeTelemetry
) -> None:
    monkeypatch.setattr(
        model_services.Request, "get", lambda *args, **kwargs: [_model_data()]
    )

    model_services.list_models(
        tasks=["OBJECT_DETECTION"],
        is_public=False,
        limit=25,
        sort="updated",
    )

    assert [event for event, *_ in fake_telemetry.events] == [
        "hubai_sdk_models_listed",
        "hubai_sdk_operation_result_recorded",
    ]
    summary_props = fake_telemetry.events[0][1]
    assert summary_props["invocation_surface"] == "python_api"
    assert summary_props["has_task_filter"] is True
    assert summary_props["visibility_filter"] == "private"
    assert summary_props["limit_bucket"] == "11_50"
    assert summary_props["result_count_bucket"] == "1_10"

    operation_props = fake_telemetry.events[1][1]
    assert operation_props["operation_name"] == "models_list"
    assert operation_props["operation_group"] == "models"
    assert operation_props["result"] == "success"


def test_create_model_suppresses_nested_get_event(
    monkeypatch: pytest.MonkeyPatch, fake_telemetry: _FakeTelemetry
) -> None:
    monkeypatch.setattr(
        model_services.Request,
        "post",
        lambda *args, **kwargs: _model_data(),
    )
    monkeypatch.setattr(
        model_services,
        "get_resource_info",
        lambda identifier, endpoint: _model_data(),
    )

    created = model_services.create_model(
        "demo-model",
        description="Demo",
        description_short="Short",
        tasks=["OBJECT_DETECTION"],
    )

    assert created.id == "aim_model"
    assert [event for event, *_ in fake_telemetry.events] == [
        "hubai_sdk_model_created",
        "hubai_sdk_operation_result_recorded",
    ]


def test_hubai_client_initialization_emits_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = _FakeTelemetry()
    monkeypatch.setattr(
        "hubai_sdk.hubai_client.get_component_telemetry",
        lambda: telemetry,
    )
    monkeypatch.setattr(
        "hubai_sdk.hubai_client.capture_client_initialized",
        lambda api_key_source: telemetry.capture(
            "hubai_sdk_client_initialized",
            {"api_key_source": api_key_source},
            include_system_metadata=True,
        ),
    )
    monkeypatch.setattr(HubAIClient, "_verify_api_key", lambda self: True)
    monkeypatch.setattr("hubai_sdk.hubai_client.load_client_plugins", dict)

    client = HubAIClient(api_key="secret")

    assert client.models is model_services
    assert [event for event, *_ in telemetry.events] == [
        "hubai_sdk_client_initialized",
        "hubai_sdk_operation_result_recorded",
    ]
    init_props = telemetry.events[0][1]
    assert init_props["api_key_source"] == "argument"
    operation_props = telemetry.events[1][1]
    assert operation_props["operation_name"] == "client_initialize"
    assert operation_props["result"] == "success"
