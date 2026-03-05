#!/usr/bin/env bash
# generate_models.sh
# Usage: ./generate_models.sh <url>

set -euo pipefail

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <url>"
  exit 1
fi

URL=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_FILE="${SCRIPT_DIR}/hubai_openapi.json"
FIXED_SCHEMA_FILE="${SCRIPT_DIR}/hubai_openapi_fixed.json"
GENERATED_FILE="${SCRIPT_DIR}/hubai_models.py"
TARGET_FILE="${REPO_ROOT}/hubai_sdk/utils/hubai_models.py"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "❌ Python not found in PATH."
  exit 1
fi

if ! command -v datamodel-codegen >/dev/null 2>&1; then
  echo "❌ datamodel-codegen not found. Install with: pip install datamodel-code-generator"
  exit 1
fi

echo "Downloading OpenAPI schema from: $URL"

CURL_ARGS=(--fail --show-error --insecure)
if [[ "${FOLLOW_REDIRECTS:-0}" == "1" ]]; then
  CURL_ARGS+=(--location)
fi
if [[ "${DEBUG:-0}" == "1" ]]; then
  CURL_ARGS+=(--verbose)
fi

curl "${CURL_ARGS[@]}" "$URL" -o "$OUTPUT_FILE"
echo "✅ Saved schema to: $OUTPUT_FILE"

(cd "$SCRIPT_DIR" && "$PYTHON_BIN" fix_nullable.py)

CODEGEN_ARGS=(
  --input "$FIXED_SCHEMA_FILE"
  --input-file-type openapi
  --output "$GENERATED_FILE"
  --target-python-version 3.10
  --output-model-type pydantic_v2.BaseModel
)

# Avoid global quote rewriting that breaks apostrophes in descriptions.
if datamodel-codegen --help 2>/dev/null | grep -q -- "--use-double-quotes"; then
  CODEGEN_ARGS+=(--use-double-quotes)
fi

# Use Annotated/StringConstraints instead of constr/conint/etc. for better type-checker support.
if datamodel-codegen --help 2>/dev/null | grep -q -- "--use-annotated"; then
  CODEGEN_ARGS+=(--use-annotated)
elif datamodel-codegen --help 2>/dev/null | grep -q -- "--field-constraints"; then
  CODEGEN_ARGS+=(--field-constraints)
fi

# Flatten root-model wrappers into direct field types where possible.
if datamodel-codegen --help 2>/dev/null | grep -q -- "--collapse-root-models"; then
  CODEGEN_ARGS+=(--collapse-root-models)
fi

# Prefer aliases over class-based RootModel wrappers for shared primitive refs.
if datamodel-codegen --help 2>/dev/null | grep -q -- "--use-type-alias"; then
  CODEGEN_ARGS+=(--use-type-alias)
elif datamodel-codegen --help 2>/dev/null | grep -q -- "--use-root-model-type-alias"; then
  CODEGEN_ARGS+=(--use-root-model-type-alias)
fi

# Keep enum-typed field defaults as enum members (helps Pylance typing).
if datamodel-codegen --help 2>/dev/null | grep -q -- "--set-default-enum-member"; then
  CODEGEN_ARGS+=(--set-default-enum-member)
fi

echo "Running datamodel-codegen with args:"
printf '  %q' datamodel-codegen "${CODEGEN_ARGS[@]}"
printf '\n'

datamodel-codegen "${CODEGEN_ARGS[@]}"

if [[ ! -s "$GENERATED_FILE" ]]; then
  echo "❌ Generated model file is empty: $GENERATED_FILE"
  exit 1
fi

# Replace AwareDatetime with NaiveDatetime
if sed --version >/dev/null 2>&1; then
  # GNU sed (Linux)
  sed -i 's/AwareDatetime/NaiveDatetime/g' "$GENERATED_FILE"
else
  # BSD sed (macOS, *BSD)
  sed -i '' 's/AwareDatetime/NaiveDatetime/g' "$GENERATED_FILE"
fi

echo "✅ Replaced all occurrences of AwareDatetime with NaiveDatetime in $GENERATED_FILE"

# Move hubai_models.py to hubai_sdk/utils/
mv "$GENERATED_FILE" "$TARGET_FILE"
echo "✅ Moved hubai_models.py to $TARGET_FILE"

rm "$FIXED_SCHEMA_FILE"
rm "$OUTPUT_FILE"
