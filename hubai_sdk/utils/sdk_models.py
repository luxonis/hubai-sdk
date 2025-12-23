from pathlib import Path

from pydantic import BaseModel

from hubai_sdk.utils.hubai_models import ModelInstanceResponse as HubAIModelInstanceResponse, ModelResponse as HubAIModelResponse, ModelVersionResponse as HubAIModelVersionResponse


class ModelResponse(HubAIModelResponse):
    pass


class ModelVersionResponse(HubAIModelVersionResponse):
    model_name: str | None = None


class ModelInstanceResponse(HubAIModelInstanceResponse):
    model_name: str | None = None
    model_variant_name: str | None = None


class ConvertResponse(BaseModel):
    downloaded_path: Path
    instance: ModelInstanceResponse
