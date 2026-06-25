#!/usr/bin/env python

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from energyevals.agent.schema import ModelSpec
from energyevals.benchmark import (
    BenchmarkConfig,
    list_questions,
    load_config,
    print_config,
    print_header,
    run_benchmark,
)
from energyevals.benchmark import (
    list_tools as list_tools_func,
)
from energyevals.benchmark.config import validate_api_keys
from energyevals.core.errors import ConfigurationError
from energyevals.core.types import ProviderName
from energyevals.tools import create_default_registry

PROVIDER_CHOICES = [str(provider) for provider in ProviderName]
SEED_MODE_CHOICES = ["fixed", "rotate", "random_per_trial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark questions against model providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                                    # Use default config
  python scripts/run_benchmark.py -c configs/my_config.yaml          # Custom config
  python scripts/run_benchmark.py --provider openrouter --model openai/gpt-4o-mini
  python scripts/run_benchmark.py --models openrouter:openai/gpt-4o-mini openrouter:anthropic/claude-sonnet-4-6
  python scripts/run_benchmark.py --list-questions                   # List questions

Config file (YAML):
  models:
    - provider: openrouter
      model: openai/gpt-4o-mini
    - provider: openrouter
      model: anthropic/claude-sonnet-4-6
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/benchmark_config.yaml"),
        help="Path to config file (default: configs/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=PROVIDER_CHOICES,
        help="Override provider from config",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Override model from config (single model mode)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specify models as provider:model pairs (e.g., openrouter:openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--questions",
        "-q",
        help="Override questions (e.g., '1,2,3' or '1-5')",
    )
    parser.add_argument(
        "--list-questions",
        "-l",
        action="store_true",
        help="List available questions and exit",
    )
    parser.add_argument(
        "--no-observe",
        action="store_true",
        help="Disable observability",
    )
    parser.add_argument(
        "--tools",
        help="Include only specific tools (comma-separated, e.g., 'battery,search,gridstatus')",
    )
    parser.add_argument(
        "--exclude-tools",
        help="Exclude specific tools (comma-separated, e.g., 'docket,tariffs')",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of independent trials per question (default: 1). "
        "Produces trial_N/ subdirectories in trace output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for deterministic shuffling when shuffle=true.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=SEED_MODE_CHOICES,
        default=None,
        help="Per-trial seed strategy when shuffle=true: fixed, rotate, or random_per_trial.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated explicit per-trial seeds (e.g., '101,202,303'). Overrides seed/seed-mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, list tools and questions, then exit without calling any LLM APIs.",
    )

    return parser.parse_args()


def apply_cli_overrides(config: BenchmarkConfig, args: argparse.Namespace) -> None:
    if args.models:
        config.models = []
        for spec in args.models:
            if ":" not in spec:
                print(f"Error: Invalid model spec '{spec}'. Use format 'provider:model'")
                sys.exit(1)
            provider, model = spec.split(":", 1)
            if provider not in PROVIDER_CHOICES:
                print(f"Error: Unknown provider '{provider}'")
                print(f"Available: {', '.join(PROVIDER_CHOICES)}")
                sys.exit(1)
            config.models.append(ModelSpec(provider=provider, model=model))
    elif args.provider or args.model:
        if len(config.models) > 1:
            print("Warning: --provider/--model overrides multi-model config to single model")
        if args.provider and not args.model:
            print("Error: --provider requires --model (no provider defaults are assumed).")
            sys.exit(1)
        provider = args.provider or config.models[0].provider
        model = args.model or config.models[0].model
        config.models = [ModelSpec(provider=provider, model=model)]

    if args.questions:
        config.questions = BenchmarkConfig.parse_questions(args.questions)

    if args.no_observe:
        config.observability_enabled = False

    if args.num_trials is not None:
        config.num_trials = args.num_trials
    if args.seed is not None:
        config.seed = args.seed
    if args.seed_mode is not None:
        config.seed_mode = args.seed_mode
    if args.seeds is not None:
        try:
            parsed_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        except ValueError:
            print("Error: --seeds must be a comma-separated list of integers")
            sys.exit(1)
        if not parsed_seeds:
            print("Error: --seeds cannot be empty")
            sys.exit(1)
        config.seeds = parsed_seeds

    if args.tools:
        config.tools_config.include = [t.strip() for t in args.tools.split(",")]
        config.tools_config.exclude = []
    elif args.exclude_tools:
        config.tools_config.exclude = [t.strip() for t in args.exclude_tools.split(",")]
        config.tools_config.include = []

    errors = config.validate()
    if errors:
        print("Error: Invalid configuration after CLI overrides:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


async def main() -> int:
    load_dotenv()

    args = parse_args()
    base_path = Path(__file__).parent.parent
    config = load_config(args.config, base_path)
    apply_cli_overrides(config, args)

    print_header("EnergyEvals Benchmark Runner")
    print_config(config)

    if args.list_tools:
        registry = create_default_registry()
        return list_tools_func(registry)

    if args.list_questions:
        return list_questions(config)

    if args.dry_run:
        try:
            validate_api_keys(config)
            key_status = "OK"
        except ConfigurationError as e:
            key_status = f"MISSING — {e}"
        print_header("Dry Run Summary")
        print(f"  Config file:   {config.config_path}")
        print(f"  Models:        {', '.join(m.display_name for m in config.models)}")
        print(f"  Questions:     {config.questions or 'all'}")
        print(f"  Max iterations:{config.max_iterations}")
        print(f"  Tool timeout:  {config.tool_timeout}s")
        print(f"  Max retries:   {config.max_retries}")
        print(f"  Retry delay:   {config.retry_base_delay}s")
        print(f"  API keys:      {key_status}")
        print("\n  Config is valid. Exiting (--dry-run).")
        return 0

    return await run_benchmark(config)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
