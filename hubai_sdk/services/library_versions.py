import re
import sys
from typing import Any
from urllib.parse import urlparse

import requests
from cyclopts import App

from hubai_sdk.errors import InputError
from hubai_sdk.utils.environ import environ
from hubai_sdk.utils.hub import raise_for_hub_error, run_cli
from hubai_sdk.utils.hub_requests import Request

app = App(
    name="library-versions",
    help="Print library versions reported by HubAI.",
    group="Admin",
)


def get_library_versions() -> dict[str, str]:
    """Get library versions exposed by the projects health endpoint."""
    try:
        response = requests.get(
            _get_library_versions_url(),
            timeout=200,
        )
        data = Request._process_response(response)
    except requests.HTTPError as exc:
        raise_for_hub_error(exc)

    return _extract_library_versions(data)


@app.default
def main() -> None:
    """Print library versions from the HubAI health endpoint."""
    versions = run_cli(get_library_versions)
    for library, version in versions.items():
        sys.stdout.write(f"{library}: {version}\n")


def _extract_library_versions(data: dict[str, Any]) -> dict[str, str]:
    versions = {
        key.removesuffix("_version").replace("_", "-"): value
        for key, value in data.items()
        if key.endswith("_version")
        and key != "version"
        and isinstance(value, str)
    }
    if not versions:
        raise InputError("No library versions found in health response.")
    return versions


def _get_library_versions_url() -> str:
    parsed_url = urlparse(environ.HUBAI_URL)
    hubai_host = parsed_url.hostname or ""
    origin = f"{parsed_url.scheme}://{parsed_url.netloc}".rstrip("/")

    match = re.fullmatch(
        r"easyml\.cloud(?:-(?P<env>[^.]+))?\.luxonis\.com",
        hubai_host,
    )
    if match:
        return f"{origin}/projects/api/v1/health/"

    return f"{origin}/api/v1/health/"
