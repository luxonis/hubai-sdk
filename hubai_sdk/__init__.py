from .hubai_client import HubAIClient
from .utils.general import version_check

__all__ = ["HubAIClient"]

__version__ = "0.1.1"

# Check for new version on PyPI
version_check(__version__)
