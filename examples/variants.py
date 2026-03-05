import os

from loguru import logger

from hubai_sdk import HubAIClient

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

# List variants
variants = client.variants.list_variants()
logger.info(f"Found {len(variants)} variants")

# List all variants of a model
model = client.models.list_models()[0]
model_id = model.id
variants = client.variants.list_variants(model_id=model_id)
logger.info(f"Found {len(variants)} variants for model {model.name}")

# Get variant by ID
variant = client.variants.get_variant(variants[0].id)
logger.info(f"Variant name: {variant.name}")
logger.info(f"Variant ID: {variant.id}")
logger.info(f"Variant description: {variant.description}")
logger.info(f"Variant version: {variant.version}")
logger.info(f"Variant platforms: {variant.platforms}")
logger.info(f"Variant exportable to: {variant.exportable_to}")
logger.info(f"Variant is public: {variant.is_public}")

# Create a new variant
new_variant = client.variants.create_variant(
    name="test-sdk-variant-py",
    model_id=model_id,
    variant_version="1.0.0",
    description="Test SDK variant",
)

logger.info(f"New variant created: {new_variant.name}")
logger.info(f"New variant ID: {new_variant.id}")
logger.info(f"New variant description: {new_variant.description}")
logger.info(f"New variant version: {new_variant.version}")
logger.info(f"New variant platforms: {new_variant.platforms}")
logger.info(f"New variant exportable to: {new_variant.exportable_to}")
logger.info(f"New variant is public: {new_variant.is_public}")

# Delete the new variant
client.variants.delete_variant(new_variant.id)
