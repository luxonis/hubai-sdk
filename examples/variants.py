import os

from hubai_sdk import HubAIClient
from hubai_sdk.services import variants

# Get API key from environment variable
api_key = os.getenv("HUBAI_API_KEY")

# Create HubAI client
client = HubAIClient(api_key=api_key)

# List variants
variants = client.variants.list_variants()
print(f"Found {len(variants)} variants\n")

# List all variants of a model
model = client.models.list_models()[0]
model_id = model.id
variants = client.variants.list_variants(model_id=model_id)
print(f"Found {len(variants)} variants for model {model.name}\n")

# Get variant by ID
variant = client.variants.get_variant(variants[0].id)
print(f"Variant name: {variant.name}")
print(f"Variant ID: {variant.id}")
print(f"Variant description: {variant.description}")
print(f"Variant version: {variant.version}")
print(f"Variant platforms: {variant.platforms}")
print(f"Variant exportable to: {variant.exportable_to}")
print(f"Variant is public: {variant.is_public}\n")

# Create a new variant
new_variant = client.variants.create_variant(
    name="test-sdk-variant-py",
    model_id=model_id,
    variant_version="1.0.0",
    description="Test SDK variant"
)

print(f"New variant created: {new_variant.name}")
print(f"New variant ID: {new_variant.id}")
print(f"New variant description: {new_variant.description}")
print(f"New variant version: {new_variant.version}")
print(f"New variant platforms: {new_variant.platforms}")
print(f"New variant exportable to: {new_variant.exportable_to}")
print(f"New variant is public: {new_variant.is_public}\n")

# Delete the new variant
client.variants.delete_variant(new_variant.id)
