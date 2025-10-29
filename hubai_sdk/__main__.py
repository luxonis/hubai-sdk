import os
import sys
import webbrowser
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

# Set a flag to indicate that the call is coming from the CLI
# we can then detect if we need to log to the console or not
os.environ["HUBAI_CALL_SOURCE"] = "CLI"


app = App(help="Interactions with resources on HubAI.", group="HubAI Commands")

app.command(model := model_app)

app.command(variant := variant_app)

app.command(instance := instance_app)

app.command(convert := cli_convert)


def validate_api_key(_: str) -> bool:
    # TODO
    return True


@app.command(group="Admin")
def login(
    relogin: Annotated[
        bool,
        Parameter(["--relogin", "-r"], help="Relogin if already logged in"),
    ] = False,
) -> None:
    """Login to HubAI.

    Parameters
    ----------
    relogin : bool
        Relogin if already logged in.
    """
    if environ.HUBAI_API_KEY and not relogin:
        logger.info(
            "User already logged in. Use `hubai --relogin` to relogin."
        )
        return

    logger.info("User not logged in. Follow the link to get your API key.")
    try:
        if not webbrowser.open("https://hub.luxonis.com/team-settings", new=2):
            logger.error(
                "Failed to open the browser. Please open the link manually: https://hub.luxonis.com/team-settings"
            )
    except Exception:
        logger.error(
            "Failed to open the browser. Please open the link manually: https://hub.luxonis.com/team-settings"
        )

    sleep(0.1)
    api_key = Prompt.ask("Enter your API key: ", password=True)
    if not validate_api_key(api_key):
        logger.error("Invalid API key. Please try again.")
        sys.exit(1)

    keyring.set_password("HubAI", "api_key", api_key)

    logger.info("API key stored successfully.")


@app.command(group="Admin")
def logout() -> None:
    """Logout from HubAI."""
    if environ.HUBAI_API_KEY is None:
        logger.info("User not logged in. Nothing to logout.")
        return

    keyring.delete_password("HubAI", "api_key")
    environ.HUBAI_API_KEY = None
    logger.info("Logged out successfully.")


if __name__ == "__main__":
    app()
