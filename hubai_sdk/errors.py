class HubApiError(Exception):
    def __init__(
        self, message: str, *, status_code: int | None = None
    ) -> None:
        self.status_code = status_code
        super().__init__(message)


class ResourceNotFoundError(HubApiError, LookupError):
    def __init__(self, identifier: str, endpoint: str) -> None:
        self.identifier = identifier
        self.endpoint = endpoint
        super().__init__(
            f"Resource for endpoint '{endpoint}' with identifier "
            f"'{identifier}' not found in HubAI.",
            status_code=404,
        )


class ResourceConflictError(HubApiError, ValueError):
    pass


class ValidationError(HubApiError, ValueError):
    pass


class InputError(ValueError):
    pass
