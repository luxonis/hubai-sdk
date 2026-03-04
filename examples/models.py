import os

from loguru import logger

from hubai_sdk import HubAIClient

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

models = client.models.list_models()
model_id = models[0].id

# Get model by ID
model = client.models.get_model(model_id)

# You can access the model attributes like this:
logger.info(f"Model name: {model.name}")
logger.info(f"Model ID: {model.id}")
logger.info(f"Model description: {model.description}")
logger.info(f"Model description short: {model.description_short}")
logger.info(f"Model tasks: {model.tasks}")

# You can also create a new model like this
new_model = client.models.create_model(
    name="test-sdk-model-py",
    license_type="MIT",
    is_public=False,
    description="Test SDK model",
    description_short="Test SDK model",
    tasks=["OBJECT_DETECTION"],
)

logger.info(f"New model created: {new_model.name}")
logger.info(f"New model ID: {new_model.id}")
logger.info(f"New model description: {new_model.description}")
logger.info(f"New model description short: {new_model.description_short}")
logger.info(f"New model tasks: {new_model.tasks}")


# You can also update the model like this
updated_model = client.models.update_model(
    new_model.id,
    license_type="Apache 2.0",
    description="Test SDK model updated with Apache 2.0 license",
    description_short="Test SDK model updated with Apache 2.0 license",
)

logger.info(f"Updated model: {updated_model.name}")
logger.info(f"Updated model ID: {updated_model.id}")
logger.info(f"Updated model description: {updated_model.description}")
logger.info(
    f"Updated model description short: {updated_model.description_short}"
)
logger.info(f"Updated model tasks: {updated_model.tasks}")

# Lastly, you can delete the models
client.models.delete_model(new_model.id)
