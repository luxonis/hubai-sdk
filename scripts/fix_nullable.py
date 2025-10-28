import json
from pathlib import Path

from loguru import logger


def fix_nullable(obj):
    if isinstance(obj, dict):
        # If property is literally `null`, replace with {"type": ["null"]}
        for k, v in list(obj.items()):
            if v is None:
                obj[k] = {"type": ["null"]}
            else:
                fix_nullable(v)

        # Fix FastAPI style "nullable: true"
        if "type" in obj and obj.get("nullable") is True:
            if isinstance(obj["type"], str):
                obj["type"] = [obj["type"], "null"]
            elif isinstance(obj["type"], list) and "null" not in obj["type"]:
                obj["type"].append("null")
            obj.pop("nullable", None)

    elif isinstance(obj, list):
        for v in obj:
            fix_nullable(v)


def main():
    input_file = Path("hubai_openapi.json")
    output_file = Path("hubai_openapi_fixed.json")

    data = json.loads(input_file.read_text())
    fix_nullable(data)
    output_file.write_text(json.dumps(data, indent=2))
    logger.info(f"✅ Fixed schema written to {output_file}")


if __name__ == "__main__":
    main()
