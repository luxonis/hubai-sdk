"""Client library for managing and converting models with HubAI.

HubAI stores model resources in a three-level hierarchy:

    1. A **model** describes the architecture, task, license, and visibility.
    2. A **variant** is a named and versioned implementation of that model.
    3. An **instance** is a concrete model artifact, such as an ONNX model or
       an exported artifact for RVC2, RVC3, RVC4, or Hailo.

`HubAIClient` is the main entry point. It validates the API key and exposes
the `hubai_sdk.services.models`, `hubai_sdk.services.variants`,
`hubai_sdk.services.instances`, and `hubai_sdk.services.convert` modules as
attributes.

Example:
    List public object-detection models and inspect the first result.

    .. python::

        from hubai_sdk import HubAIClient

        client = HubAIClient()
        models = client.models.list_models(
            tasks=["OBJECT_DETECTION"],
            is_public=True,
        )

        if models:
            model = client.models.get_model(models[0].id)
            print(model.name, model.tasks)

Authentication:
    Pass an API key to `HubAIClient`, set the ``HUBAI_API_KEY`` environment
    variable, or store a key with ``hubai login``. The explicit constructor
    argument takes precedence over the environment and stored credentials.

See Also:
    `hubai_sdk.services.convert` for hosted conversion workflows and
    `hubai_sdk.utils.sdk_models` for the response objects returned by the
    service functions.
"""

from .hubai_client import HubAIClient
from .utils.general import version_check

__all__ = ["HubAIClient"]

__version__ = "0.3.2"

# Check for new version on PyPI
version_check(__version__)
