import os
import sys
import webbrowser
from contextlib import suppress
from functools import wraps
from time import sleep
from typing import Annotated

import keyring
from cyclopts import App, Parameter
from loguru import logger
from rich.prompt import Prompt

from hubai_sdk.services.convert import convert as cli_convert
from hubai_sdk.services.instances import app as instance_app
from hubai_sdk.services.models import app as model_app
from hubai_sdk.services.variants import app as variant_app
from hubai_sdk.utils.environ import environ
from hubai_sdk.utils.hub import run_cli
from hubai_sdk.utils.plugins import load_cli_plugins
from hubai_sdk.utils.telemetry import instrument_hubai_cli

# Set a flag to indicate that the call is coming from the CLI
# we can then detect if we need to log to the console or not
os.environ["HUBAI_CALL_SOURCE"] = "CLI"

app = App(help="Interactions with resources on HubAI.", group="HubAI Commands")

app.command(model := model_app)

app.command(variant := variant_app)

app.command(instance := instance_app)


@wraps(cli_convert)
def convert_cli(*args: object, **kwargs: object) -> object:
    """Run the conversion command through the shared CLI error
    handler."""
    return run_cli(lambda: cli_convert(*args, **kwargs))


app.command(convert := convert_cli)

for plugin in load_cli_plugins():
    app.command(plugin)

instrument_hubai_cli(app)


def validate_api_key(_: str) -> bool:
    """Placeholder API key validator used by the login flow."""
    # TODO
    return True


@app.command(group="Admin")
def login(
    relogin: Annotated[
        bool,
        Parameter(["--relogin", "-r"]),
    ] = False,
) -> None:
    """Login to HubAI.

    Args:
        relogin: Relogin if already logged in.
    """
    if environ.HUBAI_API_KEY and not relogin:
        logger.info(
            "User already logged in. Use `hubai login --relogin` to relogin."
        )
        return

    logger.info("User not logged in. Follow the link to get your API key.")
    try:
        if not webbrowser.open(
            "https://hub.luxonis.com/team-settings/api-keys", new=2
        ):
            logger.warning(
                "Failed to open the browser. Please open the link manually: https://hub.luxonis.com/team-settings/api-keys"
            )
    except Exception:
        logger.warning(
            "Failed to open the browser. Please open the link manually: https://hub.luxonis.com/team-settings/api-keys"
        )

    sleep(0.1)
    api_key = Prompt.ask("Enter your API key: ", password=True)
    if not validate_api_key(api_key):
        logger.error("Invalid API key. Please try again.")
        sys.exit(1)

    try:
        keyring.set_password("HubAI", "api_key", api_key)
        logger.info("API key stored successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to store API key in keyring. Please set the HUBAI_API_KEY environment variable instead. You can do so by running `export HUBAI_API_KEY=<your_api_key>`. Error: {e}"
        )


@app.command(group="Admin")
def logout() -> None:
    """Logout from HubAI."""
    if environ.HUBAI_API_KEY is None:
        logger.info("User not logged in. Nothing to logout.")
        return

    with suppress(Exception):
        keyring.delete_password("HubAI", "api_key")
    environ.HUBAI_API_KEY = None
    logger.info("Logged out successfully.")


if __name__ == "__main__":
    app()
