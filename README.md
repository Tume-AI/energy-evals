# EnergyEvals

An evaluation framework for AI agents on energy analytics tasks. Agents use a ReAct loop over a set of energy-domain tools and are scored on a benchmark of questions.

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your keys (OPENROUTER_API_KEY is required)

# 3. Build the execution sandbox
docker build -t energyevals-sandbox -f sandbox/Dockerfile sandbox/

# 4. Run the agent interactively
python scripts/run_agent.py

# 5. Run a benchmark
python scripts/run_benchmark.py --config configs/benchmark_config.yaml
```

## Tools

All tools are optional except the agent itself. Each requires its own API key set in `.env`:

| Tool | Key | What it does |
|---|---|---|
| `search` | `EXA_API_KEY` | Web search and document retrieval |
| `gridstatus_api_tool` | `GRIDSTATUS_API_KEY` | Electricity market data (GridStatus API) |
| `tariffs` | `OPEN_EI_API_KEY` | Utility electricity rate structures (OpenEI) |
| `renewables` | `RENEWABLES_NINJA_API_KEY` | Hourly solar and wind generation profiles |
| `battery_optimization` | — | Battery revenue optimization (runs in sandbox) |
| `openweather` | `OPENWEATHER_API_KEY` | Weather data and forecasts |
| `system` | — | Sandboxed shell and Python code execution |
| Docket tools | varies | Regulatory filings (FERC, NY, TX, VA, MD, NC, SC, DC) |

### Database tool

EnergyEvals does not ship a database tool. To add your own, create a file in `energyevals/tools/` that follows this pattern:

```python
import json
from energyevals.tools.base_tool import BaseTool, tool_method

class MyDatabaseTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="database",
            description="Read-only SQL access to my energy data warehouse",
        )

    @tool_method()
    def run_query(self, query: str) -> str:
        """Run a read-only SQL query against the data warehouse.

        Args:
            query: SQL SELECT query to execute.

        Returns:
            JSON string with columns and rows.
        """
        # connect to your database and execute the query
        ...
        return json.dumps({"columns": [...], "rows": [...]})
```

Then register it in `create_default_registry()` in `energyevals/tools/__init__.py`:

```python
from energyevals.tools.my_database_tool import MyDatabaseTool
registry.register(MyDatabaseTool())
```

Any method decorated with `@tool_method()` is automatically exposed to the agent. The docstring becomes the tool description and the type hints become the JSON schema — no additional wiring needed.

## Execution Sandbox

`run_shell_command` and `run_python_code` execute inside a disposable Docker container — never on the host. The container carries no host secrets, runs as a non-root user with a read-only filesystem and resource limits, and is destroyed after each call. Battery optimization also runs inside the sandbox.

The sandbox image includes a full data-science stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `pyomo`, `ipopt`, `geopandas`, `plotly`, `xgboost`, and more.

```bash
# Build the image (required for system tool and battery optimization)
docker build -t energyevals-sandbox -f sandbox/Dockerfile sandbox/

# (Recommended) Install the egress firewall so the sandbox cannot reach your internal network
sudo ./scripts/setup_sandbox_network.sh install
echo 'SANDBOX_NETWORK=ee-sandbox' >> .env
```

Sandbox filesystem inside the container:
- `/work` — writable scratch, shared with the host's `run_outputs/` directory
- `/data` — read-only input datasets mounted from the repo's `data/` directory
- `/tmp` — per-call scratch, wiped after each call

## Benchmark

Benchmarks run a set of questions against one or more models and record per-question results, token usage, and latency.

```bash
python scripts/run_benchmark.py --config configs/benchmark_config.yaml
```

To use your own questions, create a CSV file with these columns and place it anywhere in the repo (conventionally `data/`):

| Column | Description |
|---|---|
| `S/N` | Integer row ID (1, 2, 3, ...) |
| `Category` | Topic area (e.g. `Tariffs`, `Renewables`, `Market Prices`) |
| `Question type` | Type label (e.g. `Factual`, `Analytical`, `Computational`) |
| `Difficulty level` | Difficulty label (e.g. `Easy`, `Medium`, `Hard`) |
| `Question` | The question text the agent will answer |

Then point `questions_file` in the config to your file:

```yaml
questions_file: data/my_questions.csv
```

Key config options (`configs/benchmark_config.yaml`):

Results are saved as timestamped JSON files in `results_dir`.

## Evaluation

After running a benchmark, score the results:

```bash
python scripts/run_eval.py --config configs/eval_config.yaml
```

Evaluation uses an LLM judge to score each answer and aggregates pass rates by category and difficulty.

## Architecture

**ReAct loop** — the agent iterates Thought → Action (tool call) → Observation until it produces a final answer or hits `max_iterations`.

**Provider** — all models are accessed through [OpenRouter](https://openrouter.ai), a single OpenAI-compatible endpoint that fronts OpenAI, Anthropic, Google, DeepSeek, Meta, and others.

**Tool registry** — tools self-register at init time via `@tool_method` decorators. `create_default_registry()` builds the default set; custom tools can be added without modifying framework code.

**Observability** — each run writes a JSONL trace capturing every ReAct step, tool input/output, token counts, and latency to `benchmark_traces/`.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
mypy energyevals
```
