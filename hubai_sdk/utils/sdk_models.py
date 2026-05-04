from pathlib import Path

from pydantic import BaseModel

from hubai_sdk.utils.hubai_models import (
    JobMessageResponse,
)
from hubai_sdk.utils.hubai_models import (
    ModelInstanceResponse as HubAIModelInstanceResponse,
)
from hubai_sdk.utils.hubai_models import ModelResponse as HubAIModelResponse
from hubai_sdk.utils.hubai_models import (
    ModelVersionResponse as HubAIModelVersionResponse,
)


class ModelResponse(HubAIModelResponse):
    pass


class ModelVersionResponse(HubAIModelVersionResponse):
    model_name: str | None = None


class ModelInstanceResponse(HubAIModelInstanceResponse):
    model_name: str | None = None
    model_variant_name: str | None = None


class ConvertResponse(BaseModel):
    downloaded_path: Path
    job: JobMessageResponse
    instance: ModelInstanceResponse
