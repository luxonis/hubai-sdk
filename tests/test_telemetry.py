from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

import hubai_sdk.services.convert as convert_services
import hubai_sdk.services.models as model_services
from hubai_sdk.hubai_client import HubAIClient
from hubai_sdk.utils import telemetry as telemetry_utils
from hubai_sdk.utils.types import Target

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


def test_convert_result_duration_is_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    fake_telemetry: _FakeTelemetry,
    tmp_path: Path,
) -> None:
    stage_cfg = SimpleNamespace(
        input_model=Path("model.onnx"),
        inputs=[SimpleNamespace(shape=[1, 3, 224, 224], layout="NCHW")],
    )
    cfg = SimpleNamespace(name="demo-model", stages={"main": stage_cfg})
    monotonic_values = iter([0.0, 10.0, 50.0, 60.0])

    monkeypatch.setattr(
        "hubai_sdk.utils.telemetry.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.is_nn_archive", lambda _: False
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.get_configs",
        lambda config_path, opts: (cfg, None),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.cleanup_extracted_path",
        lambda path: None,
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.normalize_quantization_input",
        lambda quantization_data: SimpleNamespace(
            quantization_data=quantization_data,
            custom_zip_path=None,
        ),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.get_target_specific_options",
        lambda target, cfg, tool_version: {},
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.create_variant",
        lambda *args, **kwargs: SimpleNamespace(id="variant-id"),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.create_instance",
        lambda *args, **kwargs: SimpleNamespace(id="instance-id"),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.upload_file",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert._export",
        lambda *args, **kwargs: SimpleNamespace(id="job-id"),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.wait_for_job",
        lambda job_id: SimpleNamespace(id=job_id),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert._resolve_exported_instance",
        lambda job: SimpleNamespace(id="exported-instance-id"),
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.download_instance",
        lambda identifier, output_dir=None: tmp_path / "artifact.blob",
    )
    monkeypatch.setattr(
        "hubai_sdk.services.convert.ConvertResponse",
        lambda **kwargs: kwargs,
    )

    convert_services.convert(
        Target.RVC4,
        path="model.onnx",
        model_id="model-id",
        variant_version="0.1.0",
    )

    conversion_result_props = next(
        properties
        for event, properties, _ in fake_telemetry.events
        if event == "hubai_sdk_conversion_result_recorded"
    )
    assert conversion_result_props["duration_ms"] == 40_000


def test_conversion_run_id_is_context_local() -> None:
    telemetry_utils.reset_conversion_run_id(None)
    main_run_id = telemetry_utils.get_or_create_conversion_run_id()
    isolated_values: dict[str, str | None] = {}

    def isolated() -> None:
        isolated_values["before"] = telemetry_utils.current_conversion_run_id()
        isolated_values["created"] = (
            telemetry_utils.get_or_create_conversion_run_id()
        )

    contextvars.Context().run(isolated)

    assert isolated_values["before"] is None
    assert isolated_values["created"] is not None
    assert isolated_values["created"] != main_run_id
    assert telemetry_utils.current_conversion_run_id() == main_run_id

    telemetry_utils.reset_conversion_run_id(None)


def test_update_model_emits_zero_updated_fields_bucket(
    monkeypatch: pytest.MonkeyPatch, fake_telemetry: _FakeTelemetry
) -> None:
    monkeypatch.setattr(
        model_services,
        "resolve_resource_id",
        lambda identifier, endpoint: "model-id",
    )
    monkeypatch.setattr(
        model_services.Request,
        "patch",
        lambda *args, **kwargs: _model_data(),
    )
    monkeypatch.setattr(
        model_services,
        "get_resource_info",
        lambda identifier, endpoint: _model_data(),
    )

    updated = model_services.update_model("demo-model")

    assert updated.id == "aim_model"
    update_props = fake_telemetry.events[0][1]
    assert fake_telemetry.events[0][0] == "hubai_sdk_model_updated"
    assert update_props["updated_field_count_bucket"] == "0"
