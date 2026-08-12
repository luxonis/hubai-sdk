"""Authenticated entry point for the HubAI Python API."""

import os
import time

from loguru import logger
from requests import HTTPError

import hubai_sdk.services.convert
import hubai_sdk.services.instances
import hubai_sdk.services.models
import hubai_sdk.services.variants
from hubai_sdk.utils.environ import environ
from hubai_sdk.utils.hub import raise_for_hub_error
from hubai_sdk.utils.hub_requests import Request
from hubai_sdk.utils.plugins import load_client_plugins
from hubai_sdk.utils.telemetry import (
    ApiKeySource,
    OperationName,
    OperationTelemetrySpec,
    TelemetryGroup,
    capture_client_initialized,
    capture_operation_result,
)


class HubAIClient:
    """Authenticate with HubAI and expose its high-level service modules.

    A client does not hold a separate HTTP session. Its service attributes are
    modules whose functions use the API key stored in the SDK environment.
    Consequently, creating another client replaces the process-wide key used
    by subsequent service calls.

    Attributes:
        models: Functions for model resources.
        variants: Functions for versioned model variants.
        instances: Functions for model artifacts and their files.
        convert: Hosted model conversion functions.

    Example:
        The constructor finds ``HUBAI_API_KEY`` automatically.

        .. python::

            from hubai_sdk import HubAIClient

            client = HubAIClient()
            models = client.models.list_models(is_public=True)

    Note:
        Installed ``hubai.plugins`` entry points can add more attributes. A
        plugin cannot replace one of the built-in service attributes.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize a HubAI SDK client.

        Args:
            api_key: HubAI API key. If omitted, the client falls back to the
                ``HUBAI_API_KEY`` environment variable and then to the key
                loaded from secure credential storage.

        Raises:
            ValueError: If no API key is available or the provided key
                is invalid.
        """
        operation_spec = OperationTelemetrySpec(
            operation_name=OperationName.CLIENT_INITIALIZE,
            operation_group=TelemetryGroup.CLIENT,
        )
        start = time.monotonic()
        api_key_source = ApiKeySource.STORED_CREDENTIALS
        caught_exc: BaseException | None = None
        # If api_key is not provided, try to get it from environment variable
        if api_key is None:
            api_key = os.getenv("HUBAI_API_KEY")
            api_key_source = ApiKeySource.ENVIRONMENT
        else:
            api_key_source = ApiKeySource.ARGUMENT

        # If still not found, try to get from environ (which may have loaded from keyring)
        if api_key is None:
            api_key = environ.HUBAI_API_KEY
            api_key_source = ApiKeySource.STORED_CREDENTIALS

        try:
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

            for attr_name, plugin in load_client_plugins().items():
                if hasattr(self, attr_name):
                    continue
                setattr(self, attr_name, plugin)

            capture_client_initialized(api_key_source)
        except BaseException as exc:
            caught_exc = exc
            raise
        finally:
            capture_operation_result(
                spec=operation_spec,
                exc=caught_exc,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    def _verify_api_key(self) -> bool:
        """Check whether the configured API key is accepted by HubAI.

        Returns:
            ``True`` if the API key is valid, otherwise ``False`` for
            authentication failures.
        """
        try:
            _ = Request.get(
                service="models",
                endpoint="models/",
                params={"is_public": False, "limit": 1},
            )
        except HTTPError as exc:
            status_code = (
                exc.response.status_code if exc.response is not None else None
            )
            if status_code in {401, 403}:
                return False
            raise_for_hub_error(exc)
        else:
            return True
