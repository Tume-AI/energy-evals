import random
import shutil

from loguru import logger

from energyevals.agent.providers import get_provider
from energyevals.agent.react_agent import ReActAgent
from energyevals.agent.schema import AgentRun, ModelSpec, ToolDefinition, ToolExecutor
from energyevals.observability import JSONFileObserver, TraceMetadata
from energyevals.tools import ToolRegistry, create_default_registry, sandbox

from energyevals.benchmark.config import BenchmarkConfig, validate_api_keys
from energyevals.benchmark.constants import QUESTION_LIST_PREVIEW_LENGTH
from energyevals.benchmark.data_loader import load_questions
from energyevals.benchmark.display import print_header, print_question, print_result
from energyevals.benchmark.models import BenchmarkResult, Question
from energyevals.benchmark.results import save_results
from energyevals.benchmark.tools import build_tool_executor, filter_tools


def _trace_run(
    observer: JSONFileObserver | None,
    run: AgentRun,
    question: Question,
    provider: str,
    model: str,
    model_spec: ModelSpec | None = None,
    tools: list[ToolDefinition] | None = None,
) -> str | None:
    """Trace an agent run through the observer if available.

    Args:
        observer: Optional observability observer
        run: The agent run to trace
        question: The benchmark question
        provider: Provider name
        model: Model name
        model_spec: Optional model specification for enriched metadata
        tools: Optional list of available tools for enriched metadata

    Returns:
        Trace ID if observer is available, None otherwise
    """
    if not observer:
        return None
    try:
        metadata: TraceMetadata = {
            "question_id": question.id,
            "category": question.category,
            "difficulty": question.difficulty,
            "provider": provider,
            "model": model,
        }
        if model_spec is not None:
            model_params: dict[str, object] = {}
            if model_spec.effort is not None:
                model_params["effort"] = model_spec.effort
            if model_spec.provider_routing is not None:
                model_params["provider_routing"] = model_spec.provider_routing
            if model_params:
                metadata["model_params"] = model_params
        if tools is not None:
            metadata["tools"] = {
                "count": len(tools),
                "names": [t.name for t in tools],
            }
        return observer.trace_agent_run(
            run=run,
            metadata=metadata,
            tags=["benchmark", question.category, question.difficulty],
        )
    except Exception as exc:
        logger.warning(f"Observer trace failed, benchmark data may be incomplete: {exc}")
        return None


async def run_question(
    agent: ReActAgent,
    question: Question,
    config: BenchmarkConfig,
    observer: JSONFileObserver | None = None,
    model_spec: ModelSpec | None = None,
    tools: list[ToolDefinition] | None = None,
) -> BenchmarkResult:
    """Run a single benchmark question.

    Args:
        agent: The ReAct agent to use
        question: The question to run
        config: Benchmark configuration
        observer: Optional observability observer
        model_spec: Model specification

    Returns:
        BenchmarkResult with metrics and answer
    """
    if model_spec is None:
        model_spec = config.models[0]

    provider = model_spec.provider
    model = model_spec.model

    try:
        run: AgentRun = await agent.run(question.question)

        metrics = {
            "input_tokens": run.total_input_tokens,
            "output_tokens": run.total_output_tokens,
            "cached_tokens": run.total_cached_tokens,
            "reasoning_tokens": run.total_reasoning_tokens,
            "total_tokens": run.total_tokens,
            "tool_calls": run.tool_calls_count,
            "iterations": run.iterations,
            "duration_seconds": run.duration_seconds,
            "latency_ms": run.total_latency_ms,
        }

        trace_id = _trace_run(observer, run, question, provider, model, model_spec=model_spec, tools=tools)

        return BenchmarkResult(
            question=question,
            provider=provider,
            model=model,
            success=run.success,
            answer=run.final_answer if config.save_answers else None,
            error=run.error,
            metrics=metrics,
            trace_id=trace_id,
        )

    except Exception as e:
        failed_run = AgentRun(
            query=question.question,
            success=False,
            error=str(e),
        )
        trace_id = _trace_run(observer, failed_run, question, provider, model, model_spec=model_spec, tools=tools)

        return BenchmarkResult(
            question=question,
            provider=provider,
            model=model,
            success=False,
            answer=None,
            error=str(e),
            trace_id=trace_id,
        )


async def _setup_observer(config: BenchmarkConfig) -> JSONFileObserver | None:
    """Set up observability observer.

    Args:
        config: Benchmark configuration

    Returns:
        Configured observer or None if disabled/failed
    """
    if not config.observability_enabled:
        return None

    config.observability_output_dir.mkdir(parents=True, exist_ok=True)
    observer = JSONFileObserver(
        output_dir=str(config.observability_output_dir),
        run_name=config.observability_run_name,
    )

    _copy_config_to_output(config)

    if config.observability_run_name:
        print(f"  Observability: JSONFileObserver (run: {config.observability_run_name})")
    else:
        print("  Observability: JSONFileObserver")
    return observer


def _copy_config_to_output(config: BenchmarkConfig) -> None:
    """Copy the benchmark config file to the output directory for traceability.

    Args:
        config: Benchmark configuration
    """
    if not config.config_path or not config.config_path.exists():
        return

    if config.observability_run_name:
        output_dir = config.observability_output_dir / config.observability_run_name
    else:
        output_dir = config.observability_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    dest_path = output_dir / "benchmark_config.yaml"
    shutil.copy2(config.config_path, dest_path)
    print(f"  Config saved to: {dest_path}")


async def _setup_tools(
    config: BenchmarkConfig,
) -> tuple[list[ToolDefinition], ToolExecutor, ToolRegistry]:
    """Set up the standard tool registry, filtered tool list, and executor.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of (tools, executor, registry). The registry is returned so the
        caller can release tool-held resources (e.g. DB pools) via ``aclose()``.
    """
    std_registry = create_default_registry()
    all_std_tools = std_registry.get_all_tools()

    tools = filter_tools(all_std_tools, config.tools_config, registry=std_registry)

    names = ", ".join(t.name for t in tools) if tools else "(none)"
    print(f"  Tools ({len(tools)}/{len(all_std_tools)}): {names}")

    executor = build_tool_executor(std_registry)

    return tools, executor, std_registry


async def _run_model_benchmark(
    model_spec: ModelSpec,
    questions: list[Question],
    tools: list[ToolDefinition],
    executor: ToolExecutor,
    config: BenchmarkConfig,
    observer: JSONFileObserver | None,
) -> list[BenchmarkResult]:
    """Run benchmark for a single model.

    Args:
        model_spec: Model specification
        questions: List of questions to run
        tools: List of available tools
        executor: Tool executor function
        config: Benchmark configuration
        observer: Optional observer

    Returns:
        List of benchmark results
    """
    print_header(f"Evaluating: {model_spec.display_name}{model_spec.params_summary}")

    provider_kwargs: dict[str, object] = {}
    if model_spec.effort:
        provider_kwargs["effort"] = model_spec.effort
    if model_spec.provider_routing:
        provider_kwargs["provider_routing"] = model_spec.provider_routing

    model_results: list[BenchmarkResult] = []

    for i, question in enumerate(questions, 1):
        print_question(question, i, len(questions))
        
        sandbox.reset_work_dir()

        provider = get_provider(model_spec.provider, model=model_spec.model, **provider_kwargs)
        agent = ReActAgent(
            provider=provider,
            tools=tools,
            tool_executor=executor,
            max_iterations=config.max_iterations,
            csv_threshold=config.csv_threshold,
            tool_timeout=config.tool_timeout,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
            max_tool_result_chars=config.max_tool_result_chars,
            tool_output_log_mode=config.tool_output_log_mode,
            tool_output_log_max_chars=config.tool_output_log_max_chars,
            tool_output_log_dir=config.tool_output_log_dir,
            tool_output_redact_secrets=config.tool_output_redact_secrets,
            history_window=config.history_window,
        )

        result = await run_question(
            agent=agent,
            question=question,
            config=config,
            observer=observer,
            model_spec=model_spec,
            tools=tools,
        )

        model_results.append(result)
        print_result(result)

    return model_results


def list_questions(config: BenchmarkConfig) -> int:
    """List available questions.

    Args:
        config: Benchmark configuration

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not config.questions_file.exists():
        print(f"Error: Questions file not found: {config.questions_file}")
        return 1

    questions = load_questions(config.questions_file)

    print_header("Available Questions")
    print(f"  File: {config.questions_file}")
    print(f"  Total: {len(questions)} questions\n")

    for q in questions:
        print(f"  {q.id}. [{q.category}] [{q.difficulty}]")
        print(f"     {q.question[:QUESTION_LIST_PREVIEW_LENGTH]}...")
        print()

    return 0


async def run_benchmark(config: BenchmarkConfig) -> int:
    """Run the benchmark with the given configuration.

    Args:
        config: Benchmark configuration

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    validate_api_keys(config)

    if not config.questions_file.exists():
        print(f"Error: Questions file not found: {config.questions_file}")
        return 1

    questions = load_questions(config.questions_file)
    if not questions:
        print(f"Error: No questions found in {config.questions_file}")
        return 1

    if config.questions:
        questions = [q for q in questions if q.id in config.questions]
        if not questions:
            print(f"Error: No matching questions for IDs: {config.questions}")
            return 1

    print(f"  Loaded {len(questions)} questions")

    trial_seeds: dict[int, int | None] = {}
    base_seed: int | None = None
    if config.shuffle and config.seeds is None and config.seed_mode in {"fixed", "rotate"}:
        base_seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)
        if config.seed is None:
            print(f"  Base shuffle seed auto-generated: {base_seed}")

    observer = await _setup_observer(config)

    tools, executor, registry = await _setup_tools(config)

    all_results: dict[str, dict[int, list[BenchmarkResult]]] = {}

    print_header("Running Multi-Model Benchmark")
    print(f"  Models: {len(config.models)}")
    for m in config.models:
        print(f"    - {m.display_name}{m.params_summary}")
    if config.num_trials > 1:
        print(f"  Trials: {config.num_trials}")

    for model_spec in config.models:
        all_results[model_spec.display_name] = {}

    try:
        for trial in range(1, config.num_trials + 1):
            if config.num_trials > 1:
                print(f"\n  --- Trial {trial}/{config.num_trials} ---")

            trial_questions = list(questions)
            trial_seed: int | None = None
            if config.shuffle:
                if config.seeds is not None:
                    trial_seed = config.seeds[trial - 1]
                elif config.seed_mode == "random_per_trial":
                    trial_seed = random.randint(0, 2**32 - 1)
                elif config.seed_mode == "fixed":
                    trial_seed = base_seed
                else:  # rotate
                    if base_seed is None:
                        base_seed = config.seed if config.seed is not None else random.randint(0, 2**32 - 1)
                    trial_seed = base_seed + (trial - 1)

                random.Random(trial_seed).shuffle(trial_questions)
                print(f"  Trial {trial} question shuffle seed: {trial_seed}")

            trial_seeds[trial] = trial_seed

            if observer:
                observer.set_trial(trial if config.num_trials > 1 else None)

            for model_spec in config.models:
                model_results = await _run_model_benchmark(
                    model_spec=model_spec,
                    questions=trial_questions,
                    tools=tools,
                    executor=executor,
                    config=config,
                    observer=observer,
                )
                all_results[model_spec.display_name][trial] = model_results

    finally:
        await registry.aclose()
        sandbox.cleanup_run()
        if observer:
            observer.set_trial(None)
            observer.flush()
            observer.shutdown()
    print_header("Summary")

    total_failed = 0
    for model_name, trials in all_results.items():
        all_model_results = [r for trial_results in trials.values() for r in trial_results]
        passed = sum(1 for r in all_model_results if r.success)
        failed = len(all_model_results) - passed
        total_failed += failed
        total_tokens = sum(r.metrics.get("total_tokens", 0) for r in all_model_results)
        total_duration = sum(r.metrics.get("duration_seconds", 0) for r in all_model_results)
        num_questions = len(next(iter(trials.values()), []))

        pct = (passed / len(all_model_results) * 100) if all_model_results else 0
        print(f"\n  {model_name}:")
        print(f"    Questions: {num_questions}")
        if config.num_trials > 1:
            print(f"    Trials: {config.num_trials}")
        print(f"    Passed: {passed} ({pct:.0f}%)")
        print(f"    Failed: {failed}")
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Total time: {total_duration:.1f}s")

    if observer:
        all_flat = [r for trials in all_results.values() for results in trials.values() for r in results]
        observer_failures = sum(1 for r in all_flat if r.trace_id is None)
        if observer_failures > 0:
            print(
                f"\n  Warning: {observer_failures} trace(s) failed to record. "
                "Check logs for details."
            )

    output_paths = save_results(all_results, config, trial_seeds=trial_seeds)
    print(f"\n  Results saved: {output_paths}")

    return 0 if total_failed == 0 else 1
