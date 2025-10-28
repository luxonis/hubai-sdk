from loguru import logger

import hubai_sdk.services.convert
import hubai_sdk.services.instances
import hubai_sdk.services.models
import hubai_sdk.services.variants
from hubai_sdk.utils.environ import environ
from hubai_sdk.utils.hub_requests import Request


class HubAIClient:
    def __init__(self, api_key: str):
        if not self._verify_api_key(api_key):
            raise ValueError("Invalid API key")
        environ.HUBAI_API_KEY = api_key
        self.models = hubai_sdk.services.models
        self.variants = hubai_sdk.services.variants
        self.instances = hubai_sdk.services.instances
        self.convert = hubai_sdk.services.convert

    def _verify_api_key(self, api_key: str) -> bool:
        try:
            _ = Request.get(
                service="models",
                endpoint="models/",
                params={"is_public": False, "limit": 1},
            )
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return False
        else:
            return True
