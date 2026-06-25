import os

from loguru import logger

from energyevals.core.errors import ConfigurationError
from energyevals.utils import HTTPClient

from energyevals.tools.base_tool import BaseTool


class APITool(BaseTool):
    """Base class for tools that call external APIs.

    Provides common functionality for API authentication, HTTP requests,
    and error handling.
    """

    def __init__(
        self,
        name: str,
        description: str,
        api_key_env_var: str,
        base_url: str | None = None,
        auth_method: str = "header",
        auth_param_name: str = "x-api-key",
        timeout: int = 30,
        retries: int = 3,
    ):
        """Initialize API tool.

        Args:
            name: Tool name
            description: Tool description
            api_key_env_var: Environment variable containing the API key
            base_url: Base URL for the API
            auth_method: Authentication method ("header" or "param")
            auth_param_name: Name of the header/param for the API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts

        Raises:
            ConfigurationError: If API key is not set
        """
        super().__init__(name, description)

        self.api_key_env_var = api_key_env_var
        self.api_key = os.getenv(api_key_env_var)

        if not self.api_key:
            logger.warning(
                f"{api_key_env_var} not set. {name} tool will not function properly."
            )

        self.base_url = base_url
        self._http = HTTPClient(
            auth_method=auth_method,
            auth_param_name=auth_param_name,
            timeout=timeout,
            retries=retries,
        )

    def require_api_key(self) -> str:
        """Get the API key, raising an error if not set.

        Returns:
            The API key

        Raises:
            ConfigurationError: If API key is not set
        """
        if not self.api_key:
            raise ConfigurationError(
                f"{self.api_key_env_var} is required for {self.name}",
                config_key=self.api_key_env_var,
            )
        return self.api_key
