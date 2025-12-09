import multiprocessing
from contextlib import suppress
import platform
import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from loguru import logger
from pydantic import model_validator
from typing_extensions import Self


def _get_password_win(
    q: multiprocessing.Queue, service_name: str, username: str
) -> None:
    try:
        result = keyring.get_password(service_name, username)
        q.put(result)
    except Exception as e:
        logger.warning(f"Failed to get password from keyring. Use the HUBAI_API_KEY environment variable instead. You can do so by running `export HUBAI_API_KEY=<your_api_key>`. Error: {e}")
        q.put(None)


def get_password_with_timeout_win(
    service_name: str, username: str, timeout: float = 5
) -> str | None:
    # if system is mac with arm, use direct keyring call
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return keyring.get_password(service_name, username)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_get_password_win, args=(q, service_name, username)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not q.empty():
        return q.get()
    return None

def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:

    # if system is mac with arm, use direct keyring call
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return keyring.get_password(service_name, username)

    def _get_password(q: multiprocessing.Queue) -> None:
        try:
            result = keyring.get_password(service_name, username)
            q.put(result)
        except Exception as e:
            logger.warning(f"Failed to get password from keyring. Use HUBAI_API_KEY from environment variable instead. You can do so by running `export HUBAI_API_KEY=<your_api_key>`. Error: {e}")
            q.put(None)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_get_password, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not q.empty():
        return q.get()
    return None


class Environ(BaseEnviron):
    HUBAI_API_KEY: str | None = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/"

    @model_validator(mode="after")
    def validate_hubai_api_key(self) -> Self:

        keyring_api_key = None

        with suppress(Exception):
            if platform.system() == "Windows":
                keyring_api_key = get_password_with_timeout_win("HubAI", "api_key")
            else:
                keyring_api_key = get_password_with_timeout("HubAI", "api_key")

        if keyring_api_key:
            if self.HUBAI_API_KEY:
                logger.warning("2 API keys found. One from environment variable and one from persistent storage (done via `hubai login`). By default, the persistent storage will be used.")
            self.HUBAI_API_KEY = keyring_api_key
            return self

        return self


environ = Environ()
