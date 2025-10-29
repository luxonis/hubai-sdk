import os
from loguru import logger

import hubai_sdk.services.convert
import hubai_sdk.services.instances
import hubai_sdk.services.models
import hubai_sdk.services.variants
from hubai_sdk.utils.environ import environ
from hubai_sdk.utils.hub_requests import Request

class HubAIClient:
    def __init__(self, api_key: str | None = None):
        # If api_key is not provided, try to get it from environment variable
        if api_key is None:
            api_key = os.getenv("HUBAI_API_KEY")

        # If still not found, try to get from environ (which may have loaded from keyring)
        if api_key is None:
            api_key = environ.HUBAI_API_KEY

        # If still not found, raise an error
        if api_key is None:
            raise ValueError(
                "API key not provided. Please provide it as a parameter, "
                "set the HUBAI_API_KEY environment variable, or use 'hubai login' "
                "to store it securely."
            )

        environ.HUBAI_API_KEY = api_key

        if not self._verify_api_key():
            raise ValueError("Invalid API key")

        logger.info("API key verified successfully.")

        self.models = hubai_sdk.services.models
        self.variants = hubai_sdk.services.variants
        self.instances = hubai_sdk.services.instances
        self.convert = hubai_sdk.services.convert

    def _verify_api_key(self) -> bool:
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
