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
    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class ModelVersionResponse(HubAIModelVersionResponse):
    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None
    model_name: str | None = None


class ModelInstanceResponse(HubAIModelInstanceResponse):
    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None
    model_name: str | None = None
    model_variant_name: str | None = None


class ModelInstanceFileResponse(HubAIModelInstanceFileResponse):
    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class JobMessageResponse(HubAIJobMessageResponse):
    team_id: ClassVar[UUID | None] = None
    user_id: ClassVar[UUID | None] = None


class ConvertResponse(BaseModel):
    downloaded_path: Path
    job: JobMessageResponse
    instance: ModelInstanceResponse
