from typing import Literal

MAX_ITERATIONS: int = 25 # Maximum number of ReAct iterations before the agent stops.

CSV_THRESHOLD: int = 20 # Row count threshold for saving tool results to CSV instead of inline.

MAX_TOKENS: int = 4096 # Default maximum tokens for LLM completion responses.

PREVIEW_ROWS: int = 5 # Number of preview rows to include when large results are saved to CSV.

QUERY_TRUNCATE_LENGTH: int = 100 # Maximum characters shown when logging a query.

TOOL_TIMEOUT: float = 120.0 # Seconds before a stalled tool call is cancelled.

PROVIDER_MAX_RETRIES: int = 3 # Maximum number of retries for provider complete() calls.

PROVIDER_RETRY_BASE_DELAY: float = 1.0 # Base delay in seconds for exponential backoff.

MAX_TOOL_RESULT_CHARS: int = 0 # Truncate tool results larger than this before adding to LLM context (0 = disabled / no truncation).

HISTORY_WINDOW: int | None = None # Max past ReAct iterations to keep in the LLM context (None or <=0 = unlimited).

TOOL_OUTPUT_LOG_MODE: Literal["off", "errors_only", "preview", "full"] = "preview"

TOOL_OUTPUT_LOG_MAX_CHARS: int = 2_000 # Max chars for console preview snippets.

TOOL_OUTPUT_LOG_DIR: str = "./run_outputs/tool_output_logs" # Directory for full tool output logs.

TOOL_OUTPUT_REDACT_SECRETS: bool = True # Redact likely secrets in console/file tool output logs.
