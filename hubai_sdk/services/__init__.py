"""High-level APIs for HubAI resources and hosted conversion.

The modules in this package form the public functional API. They are also
available through a `hubai_sdk.HubAIClient` instance:

.. list-table:: Service modules
   :header-rows: 1

   * - Module
     - Purpose
   * - `hubai_sdk.services.models`
     - Create, inspect, update, list, and delete model metadata.
   * - `hubai_sdk.services.variants`
     - Manage named and versioned implementations of a model.
   * - `hubai_sdk.services.instances`
     - Manage concrete model artifacts and transfer their files.
   * - `hubai_sdk.services.convert`
     - Upload a model, run hosted conversion, and download the result.

Most resource functions accept either a UUID or a HubAI slug. Mutating
operations raise exceptions from `hubai_sdk.errors` when the server rejects a
request; list functions return Pydantic models from
`hubai_sdk.utils.sdk_models`.
"""
