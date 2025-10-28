from json import JSONDecodeError
from typing import Any

import requests
from requests import HTTPError, Response

from hubai_sdk.typing import HubService
from hubai_sdk.utils.environ import environ


class Request:
    @staticmethod
    def url(service: HubService) -> str:
        return f"{environ.HUBAI_URL.rstrip('/')}/{service}/api/v1"

    @staticmethod
    def headers() -> dict[str, str]:
        if environ.HUBAI_API_KEY is None:
            raise ValueError("HUBAI_API_KEY is not set")

        return {
            "accept": "application/json",
            "Authorization": f"Bearer {environ.HUBAI_API_KEY}",
        }

    @staticmethod
    def _process_response(response: Response) -> Any:
        return Request._get_json(Request._check_response(response))

    @staticmethod
    def _check_response(response: Response) -> Response:
        if response.status_code >= 400:
            raise HTTPError(Request._get_json(response), response=response)
        return response

    @staticmethod
    def _get_json(response: Response) -> Any:
        try:
            return response.json()
        except JSONDecodeError as e:
            raise HTTPError(
                f"Unexpected response from the server:\n{response.text}",
                response=response,
            ) from e

    @staticmethod
    def get(service: HubService, endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.url(service)),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def post(service: HubService, endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**Request.headers(), **kwargs.pop("headers")}
        return Request._process_response(
            requests.post(
                Request._get_url(endpoint, Request.url(service)),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def delete(service: HubService, endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint, Request.url(service)),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def put(service: HubService, endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.put(
                Request._get_url(endpoint, Request.url(service)),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def patch(service: HubService, endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.patch(
                Request._get_url(endpoint, Request.url(service)),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def _get_url(endpoint: str, base_url: str | None = None) -> str:
        base_url = base_url or Request.url("models")
        return f"{base_url}/{endpoint.lstrip('/')}".rstrip("/")
