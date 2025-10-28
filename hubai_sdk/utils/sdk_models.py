from pathlib import Path

from pydantic import BaseModel

from hubai_sdk.utils.hubai_models import ModelInstanceResponse


class ConvertResponse(BaseModel):
    downloaded_path: Path
    instance: ModelInstanceResponse
