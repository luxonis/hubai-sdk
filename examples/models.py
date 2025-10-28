import os

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
print(f"Model name: {model.name}")
print(f"Model ID: {model.id}")
print(f"Model description: {model.description}")
print(f"Model description short: {model.description_short}")
print(f"Model tasks: {model.tasks}")

# You can also create a new model like this
new_model = client.models.create_model(
    name="test-sdk-model-py",
    license_type="MIT",
    is_public=False,
    description="Test SDK model",
    description_short="Test SDK model",
    tasks=["OBJECT_DETECTION"]
)

print(f"New model created: {new_model.name}")
print(f"New model ID: {new_model.id}")
print(f"New model description: {new_model.description}")
print(f"New model description short: {new_model.description_short}")
print(f"New model tasks: {new_model.tasks}")


# You can also update the model like this
updated_model = client.models.update_model(
    new_model.id,
    license_type="Apache 2.0",
    description="Test SDK model updated with Apache 2.0 license",
    description_short="Test SDK model updated with Apache 2.0 license"
)

print(f"Updated model: {updated_model.name}")
print(f"Updated model ID: {updated_model.id}")
print(f"Updated model description: {updated_model.description}")
print(f"Updated model description short: {updated_model.description_short}")
print(f"Updated model tasks: {updated_model.tasks}")

# Lastly, you can delete the models
client.models.delete_model(new_model.id)
