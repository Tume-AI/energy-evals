#!/usr/bin/env python

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from energyevals.agent import ReActAgent, get_provider
from energyevals.benchmark.tools import build_tool_executor
from energyevals.tools import create_default_registry

PROVIDERS = {
    "openrouter": "openai/gpt-4o-mini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the EnergyEvals agent a question interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_agent.py                              # interactive mode
  python scripts/run_agent.py -m anthropic/claude-sonnet-4-6   # specific model
  python scripts/run_agent.py --no-tools                   # no tools (pure LLM)
  python scripts/run_agent.py -q "What is CAISO?"          # single question
        """,
    )

    parser.add_argument(
        "--provider",
        "-p",
        choices=list(PROVIDERS.keys()),
        default="openrouter",
        help="LLM provider to use (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name (default: provider's default model)",
    )
    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="Single question to ask (skips interactive loop)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run without any tools (pure LLM mode)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum agent iterations per question (default: 15)",
    )

    return parser.parse_args()


def print_banner(provider_name: str, model_name: str, tool_count: int) -> None:
    width = 60
    print()
    print("=" * width)
    print("  EnergyEvals - Interactive Agent")
    print("=" * width)
    print(f"  Provider : {provider_name}")
    print(f"  Model    : {model_name}")
    print(f"  Tools    : {tool_count}")
    print("-" * width)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * width)
    print()


def print_answer(run) -> None:  # noqa: ANN001
    print()
    if run.success:
        print("-" * 60)
        print(run.final_answer)
        print("-" * 60)
    else:
        print(f"[Error] {run.error}")

    print(
        f"  (tokens: {run.total_tokens:,} | "
        f"tool calls: {run.tool_calls_count} | "
        f"iterations: {run.iterations} | "
        f"time: {run.duration_seconds:.1f}s)"
    )
    print()


async def run_interactive(agent: ReActAgent) -> None:
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run = await agent.run(query)
        print_answer(run)


async def main() -> int:
    load_dotenv()

    args = parse_args()
    model = args.model or PROVIDERS[args.provider]
    provider = get_provider(args.provider, model=model)

    if args.no_tools:
        tools, executor = [], None
    else:
        registry = create_default_registry()
        tools = list(registry.get_all_tools())
        executor = build_tool_executor(registry)

    agent = ReActAgent(
        provider=provider,
        tools=tools if tools else None,
        tool_executor=executor,
        max_iterations=args.max_iterations,
    )

    print_banner(args.provider, model, len(tools))

    if args.question:
        run = await agent.run(args.question)
        print_answer(run)
        return 0 if run.success else 1

    await run_interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
