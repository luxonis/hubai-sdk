"""Exceptions raised for HubAI API and user-input failures.

Service functions translate HTTP failures into this hierarchy so callers
do not need to inspect status codes. Catch `HubApiError` to handle every
remote API failure, or one of its subclasses when recovery depends on
the cause.
"""

__all__ = [
    "HubApiError",
    "InputError",
    "ResourceAmbiguousError",
    "ResourceConflictError",
    "ResourceNotFoundError",
    "ValidationError",
]


class HubApiError(Exception):
    """Base exception for errors reported by the HubAI API.

    Attributes:
        status_code: HTTP status code returned by HubAI, when available.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None
    ) -> None:
        self.status_code = status_code
        super().__init__(message)


class ResourceNotFoundError(HubApiError, LookupError):
    """Raised when a requested HubAI resource cannot be resolved.

    Attributes:
        identifier: ID, raw slug, or resource path that could not be
            resolved.
        endpoint: API resource collection that was searched.
    """

    def __init__(self, identifier: str, endpoint: str) -> None:
        self.identifier = identifier
        self.endpoint = endpoint
        super().__init__(
            f"Resource for endpoint '{endpoint}' with identifier "
            f"'{identifier}' not found in HubAI.",
            status_code=404,
        )


class ResourceAmbiguousError(HubApiError, LookupError):
    """Raised when an identifier matches more than one HubAI resource."""

    def __init__(self, identifier: str, endpoint: str) -> None:
        self.identifier = identifier
        self.endpoint = endpoint
        super().__init__(
            f"Resource for endpoint '{endpoint}' with identifier "
            f"'{identifier}' is ambiguous in HubAI. Use a more specific "
            "identifier.",
            status_code=409,
        )


class ResourceConflictError(HubApiError, ValueError):
    """Raised when an operation conflicts with an existing resource."""


class ValidationError(HubApiError, ValueError):
    """Raised when HubAI rejects request fields or values."""


class InputError(ValueError):
    """Raised when SDK input is incomplete or internally
    inconsistent."""
