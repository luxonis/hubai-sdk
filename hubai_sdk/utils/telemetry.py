import atexit
import contextvars
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any

import platform

from loguru import logger
from posthog import Posthog

from hubai_sdk.utils.general import is_cli_call, is_pip_package

# Context variable to track if telemetry should be suppressed for nested calls
_telemetry_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_telemetry_suppressed", default=False
)

class Telemetry:
    """Service for capturing anonymized telemetry data with PostHog.

    If the environment variable `HUBAI_TELEMETRY_ENABLED=False`,
    telemetry will be disabled.
    """

    PROJECT_API_KEY = "phc_jh19ssXIYbc0vD1c92xmkWCtT3uVPMmQvEy6EoBnSHY"
    HOST = "https://us.i.posthog.com"
    UNKNOWN_USER_ID = "UNKNOWN_USER_ID"

    @property
    def user_id_path(self) -> str:
        """Get the path to the user ID file."""
        return os.path.expanduser("~/.hubai/.telemetry_id")

    _curr_user_id: str | None = None

    def __init__(
        self,
        project_api_key: str | None = None,
        host: str | None = None,
    ) -> None:
        """Initialize the Telemetry instance.

        Parameters
        ----------
        project_api_key : str | None
            PostHog project API key. If None, uses the default.
        host : str | None
            PostHog host URL. If None, uses the default.
        """
        # Check if telemetry is disabled via environment variable
        telemetry_enabled = os.getenv("HUBAI_TELEMETRY_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        # Lazy import to avoid circular import
        try:
            from hubai_sdk import __version__
        except ImportError:
            __version__ = "unknown"

        self.system_metadata = {
            "cli": is_cli_call(),
            "install": "pip" if is_pip_package() else "git",
            "python": platform.python_version().rsplit(".", 1)[0],  # i.e. 3.13
            "os": platform.system(),
            "os_version": platform.release(),
            "os_arch": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "is_docker": os.path.exists("/.dockerenv"),
            "version": __version__,
        }

        if not telemetry_enabled:
            self._posthog_client = None
            logger.info("Telemetry disabled via HUBAI_TELEMETRY_ENABLED environment variable")
        else:
            logger.info(
                "Using anonymized telemetry. Set HUBAI_TELEMETRY_ENABLED=false to disable."
            )
            self._posthog_client = Posthog(
                project_api_key=project_api_key or self.PROJECT_API_KEY,
                host=host or self.HOST,
                disable_geoip=False,
                enable_exception_autocapture=True,
            )
            # Silence posthog's logging unless in debug mode
            log_level = os.getenv("HUBAI_LOG_LEVEL", "").lower()
            if log_level != "debug":
                posthog_logger = logging.getLogger("posthog")
                posthog_logger.disabled = True

    @property
    def user_id(self) -> str:
        """Get or create a consistent user ID for telemetry.

        Returns
        -------
        str
            A consistent user ID for this SDK installation
        """
        if self._curr_user_id:
            return self._curr_user_id

        # Try to get from environment variable first (for testing/override)
        distinct_id = os.getenv("HUBAI_TELEMETRY_ID")
        if distinct_id:
            self._curr_user_id = distinct_id
            return self._curr_user_id

        # File access may fail due to permissions or other reasons. We don't want to
        # crash so we catch all exceptions.
        try:
            user_id_path = self.user_id_path
            if not os.path.exists(user_id_path):
                os.makedirs(os.path.dirname(user_id_path), exist_ok=True)
                with open(user_id_path, "w") as f:
                    new_user_id = str(uuid.uuid4())
                    f.write(new_user_id)
                self._curr_user_id = new_user_id
            else:
                with open(user_id_path) as f:
                    self._curr_user_id = f.read().strip()
        except Exception:
            self._curr_user_id = self.UNKNOWN_USER_ID
            logger.debug("Failed to read/write telemetry ID, using UNKNOWN_USER_ID")

        return self._curr_user_id

    def capture(
        self,
        event_name: str,
        properties: dict[str, Any] | None = None,
        include_system_metadata: bool = False,
    ) -> None:
        """Capture a telemetry event.

        Sends events to PostHog.
        Should not be thread blocking because posthog handles it asynchronously.

        Parameters
        ----------
        event_name : str
            Name of the event to capture
        properties : dict[str, Any] | None
            Optional properties to attach to the event
        include_system_metadata: bool
            Whether to include system metadata in the event
        """
        # Check if telemetry is suppressed in the current context
        if _telemetry_suppressed.get():
            logger.debug(f"Telemetry suppressed for event: {event_name}")
            return

        if self._posthog_client is None:
            return

        if include_system_metadata:
            properties = {**self.system_metadata, **(properties or {})}

        try:
            self._posthog_client.capture(
                distinct_id=self.user_id,
                event=event_name,
                properties=properties or {},
            )
        except Exception as e:
            logger.debug(f"Failed to send Telemetry event {event_name}: {e}")

    def flush(self) -> None:
        """Flush the telemetry queue to ensure all events are sent."""
        if self._posthog_client:
            try:
                self._posthog_client.flush()
                logger.debug("PostHog client telemetry queue flushed.")
            except Exception as e:
                logger.debug(f"Failed to flush PostHog client: {e}")
        else:
            logger.debug("PostHog client not available, skipping flush.")

    def shutdown(self) -> None:
        """Shutdown the telemetry client and flush any pending
        events."""
        if self._posthog_client:
            try:
                self.flush()
                if hasattr(self._posthog_client, "shutdown"):
                    self._posthog_client.shutdown()
            except Exception as e:
                logger.debug(f"Failed to shutdown PostHog client: {e}")


# Global telemetry instances
_telemetry: Telemetry | None = None
_exit_handler_registered: bool = False


def get_telemetry() -> Telemetry | None:
    """Get the global PostHog telemetry instance.

    Returns
    -------
    Telemetry | None
        The global PostHog telemetry instance, or None if not initialized
    """
    return _telemetry


@contextmanager
def suppress_telemetry():
    """Context manager to suppress telemetry for nested function calls.

    Use this when a high-level function (like `convert`) calls other functions
    that also collect telemetry, to avoid duplicate telemetry events.

    Example:
        with suppress_telemetry():
            create_model(...)  # This won't send telemetry
            create_variant(...)  # This won't send telemetry
        # Telemetry is re-enabled after exiting the context

    Yields
    ------
    None
    """
    token = _telemetry_suppressed.set(True)
    try:
        yield
    finally:
        _telemetry_suppressed.reset(token)


def _flush_telemetry_on_exit() -> None:
    """Flush telemetry queues before program exit to ensure all events
    are sent."""
    telemetry_instance = get_telemetry()
    if telemetry_instance:
        logger.debug("Flushing Telemetry on exit")
        telemetry_instance.shutdown()

def initialize_telemetry(
    project_api_key: str | None = None,
    host: str | None = None,
) -> Telemetry:
    """Initialize the global PostHog telemetry instance (singleton
    pattern).

    Parameters
    ----------
    project_api_key : str | None
        PostHog project API key. If None, uses the default.
    host : str | None
        PostHog host URL. If None, uses the default.

    Returns
    -------
    Telemetry
        The initialized PostHog telemetry instance
    """
    global _telemetry, _exit_handler_registered
    if _telemetry is None:
        _telemetry = Telemetry(
            project_api_key=project_api_key,
            host=host,
        )
        # Register exit handler once when telemetry is first initialized
        # This ensures telemetry is flushed on exit for both CLI and Python API usage
        if not _exit_handler_registered:
            atexit.register(_flush_telemetry_on_exit)
            _exit_handler_registered = True
            logger.debug("Exit handler for Telemetry registered successfully.")

    logger.debug("Telemetry initialized successfully.")
    return _telemetry
