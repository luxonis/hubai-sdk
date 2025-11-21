import multiprocessing
from contextlib import suppress
import platform
import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from loguru import logger
from pydantic import model_validator
from typing_extensions import Self


def _get_password(
    q: multiprocessing.Queue, service_name: str, username: str
) -> None:
    try:
        result = keyring.get_password(service_name, username)
        q.put(result)
    except Exception as e:
        logger.error(f"Failed to get password: {e}")
        q.put(None)


def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:
    # if system is mac with arm, use direct keyring call
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return keyring.get_password(service_name, username)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_get_password, args=(q, service_name, username)
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


class Environ(BaseEnviron):
    HUBAI_API_KEY: str | None = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/"

    @model_validator(mode="after")
    def validate_hubai_api_key(self) -> Self:

        with suppress(Exception):
            keyring_api_key = get_password_with_timeout("HubAI", "api_key")

        if keyring_api_key:
            if self.HUBAI_API_KEY:
                logger.warning("2 API keys found. One from environment variable and one from persistent storage (done via `hubai login`). By default, the persistent storage will be used.")
            self.HUBAI_API_KEY = keyring_api_key
            return self

        return self


environ = Environ()
