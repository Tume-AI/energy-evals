from typing import Any

from energyevals.utils import HTTPClient


class HTTPMixin:
    """Mixin for tools that need HTTP request capabilities.

    Provides a configured HTTP client for making API requests.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._http_client: HTTPClient | None = None

    def get_http_client(
        self,
        auth_method: str = "header",
        auth_param_name: str = "x-api-key",
        timeout: int = 30,
        retries: int = 3,
    ) -> HTTPClient:
        """Get or create HTTP client.

        Args:
            auth_method: Authentication method ("header" or "param")
            auth_param_name: Name of the header/param for the API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts

        Returns:
            Configured HTTP client
        """
        if self._http_client is None:
            self._http_client = HTTPClient(
                auth_method=auth_method,
                auth_param_name=auth_param_name,
                timeout=timeout,
                retries=retries,
            )
        return self._http_client
