#!/usr/bin/env bash
# generate_models.sh
# Usage: ./generate_models.sh <url>

URL=$1
OUTPUT_FILE="hubai_openapi.json"

echo "Downloading OpenAPI schema from: $URL"

if curl -sSL "$URL" -o "$OUTPUT_FILE"; then
    echo "✅ Saved schema to: $OUTPUT_FILE"
else
    echo "❌ Failed to download schema from $URL"
fi

python fix_nullable.py

datamodel-codegen \
  --input hubai_openapi_fixed.json \
  --input-file-type openapi \
  --output hubai_models.py \
  --target-python-version 3.12 \
  --output-model-type pydantic_v2.BaseModel


# Replace AwareDatetime with NaiveDatetime
# Detect OS type
if sed --version >/dev/null 2>&1; then
  # GNU sed (Linux)
  sed -i 's/AwareDatetime/NaiveDatetime/g' hubai_models.py
else
  # BSD sed (macOS, *BSD)
  sed -i '' 's/AwareDatetime/NaiveDatetime/g' hubai_models.py
fi

echo "✅ Replaced all occurrences of AwareDatetime with NaiveDatetime in hubai_models.py"

# Move hubai_models.py to hubai_sdk/utils/
mv hubai_models.py ../hubai_sdk/utils/

echo "✅ Moved hubai_models.py to hubai_sdk/utils/"

rm hubai_openapi_fixed.json
rm hubai_openapi.json
