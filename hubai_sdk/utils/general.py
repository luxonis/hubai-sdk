import os
import re
from pathlib import Path

from loguru import logger

import sys
import requests
from packaging.version import parse as parse_version, Version

PYPI_URL = "https://pypi.org/pypi/hubai-sdk/json"


def _normalize_underscores(s: str) -> str:
    return re.sub(r"_+", "_", s)


def sanitize_net_name(name: str, with_suffix: bool = False) -> str:
    """Sanitize net name or path. If input is a path, only sanitize the
    basename. If input is a name, sanitize the whole string. Collapse
    multiple underscores.

    @type name: str
    @param name: The name or path to sanitize.
    @type with_suffix: bool
    @param with_suffix: If True, the suffix (file extension) is
        preserved and not sanitized.
    """
    p = Path(name)
    base, stem, suffix = p.name, p.stem, p.suffix

    if len(p.parts) > 1:
        if with_suffix and suffix:
            sanitized_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
            sanitized_stem = _normalize_underscores(sanitized_stem)
            sanitized_base = sanitized_stem + suffix
        else:
            sanitized_base = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
            sanitized_base = _normalize_underscores(sanitized_base)
        if sanitized_base != p.name:
            logger.warning(
                f"Illegal characters detected in: '{p.name}'. Replacing with '_'. New name: '{sanitized_base}'"
            )
        return (
            str(p.parent / sanitized_base)
            if str(p.parent) != "."
            else sanitized_base
        )

    if with_suffix and suffix:
        sanitized_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
        sanitized_stem = _normalize_underscores(sanitized_stem)
        sanitized = sanitized_stem + suffix
    else:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        sanitized = _normalize_underscores(sanitized)
    if sanitized != name:
        logger.warning(
            f"Illegal characters detected in: '{name}'. Replacing with '_'. New name: '{sanitized}'"
        )
    return sanitized


def is_cli_call() -> bool:
    return os.environ.get("HUBAI_CALL_SOURCE") == "CLI"


def is_pip_package(filepath: str = __name__) -> bool:
    """Determine if the file at the given filepath is part of a pip
    package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None


def significant_update_available(cur: Version, new: Version) -> bool:
    # Must be newer at all
    if new <= cur:
        return False

    # Notify on major or minor change only
    if new.major != cur.major:
        return True

    if new.minor != cur.minor:
        return True

    # Patch-only change is ignored
    return False


def version_check(current_version: str):
    try:
        response = requests.get(PYPI_URL, timeout=2)
        data = response.json()

        latest = parse_version(data["info"]["version"])
        current = parse_version(current_version)

        logger.info(f"Latest version: {latest}, Current version: {current}")

        if significant_update_available(current, latest):
            logger.error(
                f"A newer version of hubai-sdk is available: {latest} "
                f"(current: {current}).\n"
                f"Run: `pip install --upgrade hubai-sdk`",
                file=sys.stderr,
            )
            exit(1)

    except Exception:
        # Never crash user code
        pass
