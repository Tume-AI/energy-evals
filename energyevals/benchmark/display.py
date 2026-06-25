from energyevals.benchmark.config import BenchmarkConfig
from energyevals.benchmark.constants import ANSWER_PREVIEW_LENGTH, HEADER_WIDTH, QUESTION_PREVIEW_LENGTH
from energyevals.benchmark.models import BenchmarkResult, Question


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * HEADER_WIDTH}")
    print(f"  {text}")
    print(f"{'=' * HEADER_WIDTH}")


def print_config(config: BenchmarkConfig) -> None:
    """Print configuration summary."""
    print("\n  Configuration:")
    print(f"    Models ({len(config.models)}):")
    for m in config.models:
        print(f"      - {m.display_name}{m.params_summary}")
    print(f"    Questions file: {config.questions_file}")
    print(f"    Questions: {config.questions or 'all'}")
    print(f"    Observability: {'enabled' if config.observability_enabled else 'disabled'}")
    print(f"    Max iterations: {config.max_iterations}")


def print_question(q: Question, index: int, total: int) -> None:
    """Print question details."""
    print(f"\n  [{index}/{total}] Question {q.id} | {q.category} | {q.difficulty}")
    print(f"  {q.question[:QUESTION_PREVIEW_LENGTH]}..." if len(q.question) > QUESTION_PREVIEW_LENGTH else f"  {q.question}")


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result."""
    status = "[PASS]" if result.success else "[FAIL]"
    print(f"\n  {status}")

    if result.answer:
        answer_preview = (
            result.answer[:ANSWER_PREVIEW_LENGTH] + "..." if len(result.answer) > ANSWER_PREVIEW_LENGTH else result.answer
        )
        print(f"  Answer: {answer_preview}")

    if result.error:
        print(f"  Error: {result.error}")

    if result.metrics:
        print(
            f"  Metrics: tokens={result.metrics.get('total_tokens', 0)}, "
            f"tools={result.metrics.get('tool_calls', 0)}, "
            f"time={result.metrics.get('duration_seconds', 0):.1f}s"
        )

    if result.trace_id:
        print(f"  Trace: {result.trace_id}")
