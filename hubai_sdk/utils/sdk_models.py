"""Stable response models returned by the public SDK services.

The HubAI OpenAPI schema is generated into an internal module. The
wrappers here remove server-only ownership fields and add values
populated by the SDK, providing a smaller and more accurate public
contract.

All response types are Pydantic models. Use normal attribute access for
fields or ``model_dump()`` when a dictionary is required.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel

from hubai_sdk.utils.hubai_models import (
    JobMessageResponse as HubAIJobMessageResponse,
)
from hubai_sdk.utils.hubai_models import (
    ModelInstanceFileResponse as HubAIModelInstanceFileResponse,
)
from hubai_sdk.utils.hubai_models import (
    ModelInstanceResponse as HubAIModelInstanceResponse,
)
from hubai_sdk.utils.hubai_models import ModelResponse as HubAIModelResponse
from hubai_sdk.utils.hubai_models import (
    ModelVersionResponse as HubAIModelVersionResponse,
)


# The generated OpenAPI models include team_id/user_id because they are present
# in the internal Pydantic response models. The public API middleware strips
# those fields from real client responses, so the SDK-facing wrappers hide them
# to match the actual external contract.
class ModelResponse(HubAIModelResponse):
    """Metadata for a top-level HubAI model resource.

    Common fields:
        - ``id``, ``name``, and ``slug`` identify the model.
        - ``tasks``, ``license_type``, and ``is_yolo`` describe its contract.
        - ``is_public`` reports its visibility.
        - ``platforms`` and ``exportable_to`` summarize artifact support.
        - ``versions``, ``likes``, and ``downloads`` contain aggregate counts.
    """

    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class ModelVersionResponse(HubAIModelVersionResponse):
    """Metadata for a versioned model variant.

    Common fields:
        - ``id``, ``name``, ``slug``, and ``variant_slug`` identify the
          variant.
        - ``model_id`` identifies its parent model.
        - ``version`` is the semantic model version.
        - ``platforms`` and ``exportable_to`` describe available artifacts and
          possible export targets.

    Attributes:
        model_name: Parent model name when requested by the caller. Some list
            operations leave this as ``None`` to avoid an extra API request.
    """

    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None
    model_name: str | None = None


class ModelInstanceResponse(HubAIModelInstanceResponse):
    """Metadata for a concrete model artifact.

    Common fields:
        - ``id``, ``name``, and ``slug`` identify the artifact.
        - ``model_id`` and ``model_version_id`` identify its parents.
        - ``model_type``, ``model_precision_type``, and ``input_shape``
          describe the stored model.
        - ``parent_id`` links an exported artifact to its source instance.
        - ``status`` reports whether the artifact is ready for use.

    Attributes:
        model_name: Parent model name when explicitly requested.
        model_variant_name: Parent variant name when explicitly requested.
    """

    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None
    model_name: str | None = None
    model_variant_name: str | None = None


class ModelInstanceFileResponse(HubAIModelInstanceFileResponse):
    """Metadata for one file attached to a model instance.

    Common fields:
        ``id`` identifies the file record, ``model_instance_id`` identifies its
        owner, ``filepath`` is relative to the artifact root, and
        ``file_size_bytes`` is its stored size.
    """

    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class JobMessageResponse(HubAIJobMessageResponse):
    """Status and output metadata for an asynchronous HubAI job.

    Common fields:
        ``id``, ``name``, and ``status`` identify the job and its state.
        ``queued_at``, ``started_at``, and ``finished_at`` describe timing.
        Successful jobs populate ``result``; failed jobs may populate
        ``exception`` and ``logs``.
    """

    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class ConvertResponse(BaseModel):
    """Result of a completed hosted model conversion.

    Attributes:
        downloaded_path: Local directory or file created by the downloader.
        job: Completed export job returned by HubAI.
        instance: Exported model instance associated with the job.
    """

    downloaded_path: Path
    job: JobMessageResponse
    instance: ModelInstanceResponse
