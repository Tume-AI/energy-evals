HTTP_TIMEOUT_EXTENDED: int = 60 # Extended timeout for slower APIs (GridStatus, FERC, etc.).

HTTP_TIMEOUT_LONG: int = 120 # Long timeout for very slow APIs (Renewables.ninja).

RENEWABLES_MIN_REQUEST_INTERVAL_S: float = 1.1 # Min seconds between Renewables.ninja requests (burst limit is 1/second).

RENEWABLES_MAX_RETRIES: int = 4 # Retries on a Renewables.ninja 429 before giving up.

RENEWABLES_RETRY_BASE_DELAY_S: float = 1.0 # Base delay (s) for Renewables.ninja 429 backoff when no Retry-After header.

SEARCH_MAX_RESULTS: int = 8 # Default and enforced per-call ceiling on search results returned.

SEARCH_TEXT_MAX_CHARS: int = 8000 # Per-result page-text cap sent to Exa (text.maxCharacters; Exa hard max is 10000).

SEARCH_HIGHLIGHTS_MAX_CHARS: int = 2000 # Per-result highlights cap sent to Exa (highlights.maxCharacters; Exa hard max is 10000).

SYSTEM_MAX_RESULTS: int = 200 # Default maximum results for file listing and grep operations.

GREP_MAX_LINE_CHARS: int = 300 # Truncate each grep match line to this many chars (a match in a minified/data file can be huge).

GREP_MAX_TOTAL_CHARS: int = 100_000 # Hard ceiling on total grep_files output; trailing matches are dropped past this.

GREP_EXCLUDE_DIRS: tuple[str, ...] = (
    ".git", ".venv", ".mypy_cache", ".ruff_cache", "__pycache__", ".pytest_cache", "node_modules",
)  # Directories grep_files never searches (caches/vendored; never answer-relevant).

SYSTEM_COMMAND_TIMEOUT: int = 60 # Default timeout in seconds for shell command execution.

BATTERY_ROUNDING_PROFILE: int = 4 # Decimal places for battery operation profile values.

BATTERY_ROUNDING_SUMMARY: int = 2 # Decimal places for battery summary revenue metrics.

BATTERY_INITIAL_SOC_FRACTION: float = 0.5 # Initial state of charge as fraction of capacity (0.5 = 50%).

DATA_PREVIEW_SIZE: int = 10 # Number of sample data items to include in API response previews.

CSV_PREVIEW_ROWS: int = 5 # Number of rows to preview when displaying CSV data.

BATTERY_CSV_MAX_FILE_SIZE_MB: int = 200  # Maximum input CSV size for battery optimization.
BATTERY_CSV_MAX_ROWS: int = 10_000_000      # Maximum rows loaded from battery optimization CSV.

GRID_STATUS_PAGE_SIZE: int = 50_000  # Maximum rows per GridStatus API query (API hard limit).
