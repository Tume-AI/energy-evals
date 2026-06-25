import os
import re
import ssl
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import certifi
import requests
from loguru import logger

from energyevals.utils.constants import HTTP_RETRIES_DEFAULT, HTTP_TIMEOUT_DEFAULT

# Statuses worth retrying: 429 (rate limit) + transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Query-string params that carry credentials. When a request fails, the raised
# exception's str() and the response body can echo the full request URL --
# including these -- which then lands in the tool output and the agent trace.
# Redact them before any error text is returned to the model.
_SECRET_QUERY_PARAMS = (
    "appid", "api_key", "apikey", "access_token", "token",
    "secret", "password", "pwd", "auth", "key", "sig", "signature",
)
_SECRET_QS_RE = re.compile(
    r"(?i)([?&](?:" + "|".join(_SECRET_QUERY_PARAMS) + r")=)[^&\s\"'\\]+"
)


def redact_url_secrets(text: str) -> str:
    """Mask credential query-params (``appid=``, ``token=``, ...) in any string.

    Used on HTTP error text before it is surfaced to the agent/trace, so an
    upstream error that echoes the request URL cannot leak an API key.
    """
    if not text:
        return text
    return _SECRET_QS_RE.sub(r"\1REDACTED", text)


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def request_with_retry(
    method: str,
    url: str,
    *,
    retries: int = 2,
    base_delay: float = 1.0,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> requests.Response:
    """HTTP request with retry on 429 (honoring ``Retry-After``) and transient 5xx/timeouts.

    Returns the final ``Response`` (the caller checks status / parses it); re-raises
    the last connection/timeout error if every attempt fails. Does NOT throttle --
    use for APIs with generous limits where only transient failures need handling
    (for tight per-second limits, add a throttle as GridStatus/Renewables do).
    """
    caller = session or requests
    do = getattr(caller, method.lower())
    response: requests.Response | None = None
    for attempt in range(retries + 1):
        try:
            response = do(url, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries:
                time.sleep(base_delay * (2**attempt))
                continue
            raise
        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            delay = _retry_after_seconds(response) or base_delay * (2**attempt)
            logger.warning(
                f"HTTP {response.status_code} from {url}; retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{retries + 1})"
            )
            time.sleep(delay)
            continue
        return response
    return response  # type: ignore[return-value]


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    retries: int = 2,
    base_delay: float = 1.0,
) -> T:
    """Call ``fn`` (e.g. an SDK call with no ``Response`` object) and retry with
    exponential backoff while ``is_retryable(exc)`` is True (e.g. a rate-limit error)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt < retries and is_retryable(exc):
                time.sleep(base_delay * (2**attempt))
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover

# Well-known system CA bundle paths by platform.
_SYSTEM_CA_PATHS = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian / Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL / CentOS / Fedora
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/etc/pki/tls/cacert.pem",  # OpenELEC
    "/etc/ssl/cert.pem",  # macOS / Alpine
)


def http_error_detail(exc: requests.exceptions.RequestException) -> str:
    """Extract the API's own error message from a failed request's response body.

    ``str(exc)`` only carries the status line (e.g. "400 Client Error: Bad
    Request for url: ..."); the response body usually explains *why* (e.g.
    "date_to exceeds allowed max_daterange of P1Y", or a GridStatus quota
    message). Returns "" when there is no response (connection errors) or no
    usable body, so callers can fall back to ``str(exc)``.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        body = response.json()
    except ValueError:
        return redact_url_secrets((response.text or "").strip()[:300])
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            value = body.get(key)
            if value:
                return redact_url_secrets(str(value))
    return redact_url_secrets(str(body)[:300])


@lru_cache(maxsize=1)
def get_system_ca_bundle() -> str:
    """Return the best available CA certificate bundle path.

    Prefers the system CA store (which typically carries CAs that the
    ``certifi`` package may have removed) and falls back to ``certifi``
    if no system bundle is found.
    """
    # Python's compiled-in OpenSSL default comes first.
    openssl_cafile = ssl.get_default_verify_paths().openssl_cafile
    if openssl_cafile and os.path.isfile(openssl_cafile):
        return openssl_cafile

    for path in _SYSTEM_CA_PATHS:
        if os.path.isfile(path):
            return path

    return certifi.where()


class HTTPClient:
    """Simple HTTP client for API requests with common authentication patterns.

    Provides a unified interface for making HTTP requests with different
    authentication methods (header-based, parameter-based).
    """

    def __init__(
        self,
        auth_method: str = "header",
        auth_param_name: str = "x-api-key",
        timeout: int = HTTP_TIMEOUT_DEFAULT,
        retries: int = HTTP_RETRIES_DEFAULT,
    ):
        """Initialize HTTP client.

        Args:
            auth_method: Authentication method ("header" or "param")
            auth_param_name: Name of the header/param for the API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        self.auth_method = auth_method
        self.auth_param_name = auth_param_name
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def get(
        self,
        url: str,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make a GET request.

        Args:
            url: URL to request
            api_key: API key for authentication
            params: Query parameters
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: If request fails
        """

        params = params or {}
        headers = headers or {}

        if api_key:
            if self.auth_method == "header":
                headers[self.auth_param_name] = api_key
            elif self.auth_method == "param":
                params[self.auth_param_name] = api_key

        last_error: requests.exceptions.RequestException | None = None
        for attempt in range(1 + self.retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    delay = 2 ** attempt
                    logger.warning(f"HTTP GET failed (attempt {attempt + 1}/{1 + self.retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
        logger.error(f"HTTP request failed after {1 + self.retries} attempts: {last_error}")
        raise last_error  # type: ignore[misc]

    def post(
        self,
        url: str,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a POST request.

        Args:
            url: URL to request
            api_key: API key for authentication
            params: Query parameters
            headers: Additional headers
            json_data: JSON data to send in body

        Returns:
            Parsed JSON response

        Raises:
            requests.exceptions.RequestException: If request fails
        """

        params = params or {}
        headers = headers or {}

        if api_key:
            if self.auth_method == "header":
                headers[self.auth_param_name] = api_key
            elif self.auth_method == "param":
                params[self.auth_param_name] = api_key

        last_error: requests.exceptions.RequestException | None = None
        for attempt in range(1 + self.retries):
            try:
                response = self.session.post(
                    url,
                    params=params,
                    headers=headers,
                    json=json_data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    delay = 2 ** attempt
                    logger.warning(f"HTTP POST failed (attempt {attempt + 1}/{1 + self.retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
        logger.error(f"HTTP request failed after {1 + self.retries} attempts: {last_error}")
        raise last_error  # type: ignore[misc]
