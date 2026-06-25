import asyncio
import concurrent.futures
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from energyevals.core.retry import retry_with_backoff

from energyevals.agent.constants import (
    CSV_THRESHOLD,
    HISTORY_WINDOW,
    MAX_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    PROVIDER_MAX_RETRIES,
    PROVIDER_RETRY_BASE_DELAY,
    QUERY_TRUNCATE_LENGTH,
    TOOL_OUTPUT_LOG_DIR,
    TOOL_OUTPUT_LOG_MAX_CHARS,
    TOOL_OUTPUT_LOG_MODE,
    TOOL_OUTPUT_REDACT_SECRETS,
    TOOL_TIMEOUT,
)
from energyevals.agent.exceptions import ContextWindowExceededError, ProviderError, ToolExecutionError
from energyevals.agent.processors import ResultProcessor
from energyevals.agent.prompts import get_system_prompt
from energyevals.agent.providers import BaseProvider, ProviderResponse, ToolDefinition
from energyevals.agent.schema import (
    AgentRun,
    AgentStep,
    ImageContent,
    Message,
    StepType,
    TextContent,
    ToolExecutor,
)

_RAW_TOOL_CALL_RE = re.compile(r"<function=\w+.*?</function>", re.DOTALL)
_SECRET_JSON_VALUE_RE = re.compile(
    r'(?i)("?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|authorization)"?\s*:\s*")([^"]+)(")'
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|authorization)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9\-._~+/]+=*)")


class ReActAgent:
    """Custom ReAct agent with multi-provider support.

    This agent implements the ReAct (Reasoning and Acting) pattern,
    alternating between thinking about the problem and taking actions
    via tools to gather information.
    """

    def __init__(
        self,
        provider: BaseProvider,
        tools: list[ToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = MAX_ITERATIONS,
        system_prompt: str | None = None,
        csv_threshold: int = CSV_THRESHOLD,
        csv_output_dir: str = "./run_outputs",
        result_processor: ResultProcessor | None = None,
        tool_timeout: float = TOOL_TIMEOUT,
        max_retries: int = PROVIDER_MAX_RETRIES,
        retry_base_delay: float = PROVIDER_RETRY_BASE_DELAY,
        max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
        tool_output_log_mode: str = TOOL_OUTPUT_LOG_MODE,
        tool_output_log_max_chars: int = TOOL_OUTPUT_LOG_MAX_CHARS,
        tool_output_log_dir: str | Path = TOOL_OUTPUT_LOG_DIR,
        tool_output_redact_secrets: bool = TOOL_OUTPUT_REDACT_SECRETS,
        history_window: int | None = HISTORY_WINDOW,
    ):
        """Initialize the ReAct agent.

        Defaults for the tunable arguments live in ``.constants``; only the
        arguments with non-obvious semantics are documented here.

        Args:
            provider: The LLM provider to use.
            tools: List of available tools.
            tool_executor: Function to execute tool calls. Defaults to one that errors.
            system_prompt: Custom system prompt. If None, uses default prompt.
            result_processor: Custom result processor. If None, creates default.
            max_tool_result_chars: Truncate tool results to this many chars before adding to LLM context (0 = disabled).
            tool_output_log_mode: Tool output logging mode: off, errors_only, preview, or full.
            tool_output_log_dir: Directory where full tool outputs are saved in full mode.
            tool_output_redact_secrets: Whether likely secrets are redacted in console/file logs.
            history_window: Maximum past ReAct iterations to retain in the LLM
                context. ``None`` or values <= 0 mean unlimited (no trimming).
                The system message and the original user query are always kept;
                trimming drops whole assistant + tool_result groups to preserve
                tool_call_id pairing required by OpenAI-compatible APIs.
        """
        self.provider = provider
        self.tools = tools or []
        self.tool_executor = tool_executor or self._default_tool_executor
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or get_system_prompt()
        self._result_processor = result_processor or ResultProcessor(
            csv_threshold=csv_threshold,
            csv_output_dir=csv_output_dir,
        )
        self._tool_registry: dict[str, ToolDefinition] = {t.name: t for t in self.tools}
        self.tool_timeout = tool_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_tool_result_chars = max_tool_result_chars
        self.tool_output_log_mode = tool_output_log_mode
        self.tool_output_log_max_chars = tool_output_log_max_chars
        self.tool_output_log_dir = Path(tool_output_log_dir)
        self.tool_output_redact_secrets = tool_output_redact_secrets
        self.history_window = history_window

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tools.append(tool)
        self._tool_registry[tool.name] = tool

    def register_tools(self, tools: list[ToolDefinition]) -> None:
        for tool in tools:
            self.register_tool(tool)

    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Execute the ReAct loop for a given query.

        Args:
            query: The user's query to process.
            context: Optional additional context to include.

        Returns:
            AgentRun containing the full execution trace and result.
        """
        run = AgentRun(query=query)
        messages = self._build_initial_messages(query, context)

        logger.debug(f"Starting agent run for query: {query[:QUERY_TRUNCATE_LENGTH]}...")

        try:
            for iteration in range(self.max_iterations):
                run.iterations = iteration + 1

                response = await self._get_response(messages, run)

                run.total_input_tokens += response.input_tokens
                run.total_cached_tokens += response.cached_tokens
                run.total_output_tokens += response.output_tokens
                run.total_reasoning_tokens += response.reasoning_tokens
                run.total_latency_ms += response.latency_ms

                if response.tool_calls:
                    should_continue = await self._process_tool_calls(
                        response, messages, run, iteration
                    )
                    if not should_continue:
                        break
                else:
                    answer = response.content or ""
                    if not answer.strip():
                        # No tool calls and no answer text -- a degenerate turn.
                        # Don't let an empty response masquerade as a successful
                        # final answer; nudge the model and continue the loop.
                        logger.warning(
                            "Model returned an empty response with no tool calls "
                            f"(iteration {iteration}); nudging it to act or answer."
                        )
                        run.steps.append(
                            AgentStep(
                                step_type=StepType.THOUGHT,
                                content="[empty model response -- nudged to continue]",
                                iteration=iteration,
                                tokens_used=response.input_tokens + response.output_tokens,
                                latency_ms=response.latency_ms,
                                reasoning=response.reasoning_content,
                            )
                        )
                        messages.append(
                            Message(
                                role="user",
                                content=(
                                    "Your last response was empty. You must either "
                                    "call a tool to make progress, or state your "
                                    "final answer."
                                ),
                            )
                        )
                        continue
                    if _RAW_TOOL_CALL_RE.search(answer):
                        logger.warning(
                            "Model returned raw text tool call(s) instead of structured tool_calls. "
                            "The provider may not have parsed them. "
                            f"Content preview: {answer[:200]}"
                        )
                    run.final_answer = answer
                    run.steps.append(
                        AgentStep(
                            step_type=StepType.ANSWER,
                            content=answer,
                            iteration=iteration,
                            tokens_used=response.input_tokens + response.output_tokens,
                            latency_ms=response.latency_ms,
                            reasoning=response.reasoning_content,
                        )
                    )
                    run.success = True
                    break

            if not run.success and run.iterations >= self.max_iterations:
                run.error = f"Max iterations ({self.max_iterations}) reached"
                logger.warning(run.error)

        except Exception as e:
            run.error = str(e)
            run.steps.append(
                AgentStep(
                    step_type=StepType.ERROR,
                    content=str(e),
                )
            )
            logger.error(f"Agent run failed: {e}")

        run.end_time = time.time()
        logger.info(
            f"Agent run completed: success={run.success}, "
            f"iterations={run.iterations}, tokens={run.total_tokens}"
        )

        return run

    def _build_initial_messages(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> list[Message]:
        # Prompt-cache breakpoints are (re)applied before every provider call
        # by _apply_cache_breakpoints(), so none are set on the messages here.
        # Tool name + description + schema are sent through the provider's
        # structured `tools=` parameter on every call, so the system prompt
        # deliberately does not duplicate them as a natural-language list.
        messages = [
            Message(role="system", content=self.system_prompt),
        ]

        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            messages.append(
                Message(
                    role="user",
                    content=f"Context:\n{context_str}\n\nQuery: {query}",
                )
            )
        else:
            messages.append(Message(role="user", content=query))

        return messages

    def _apply_cache_breakpoints(self, messages: list[Message]) -> None:
        """(Re)place prompt-cache breakpoints right before a provider call.

        Two breakpoints are maintained:

        - a *static* one on the system message, which caches the system
          prompt plus the tool schemas (the schemas precede the system
          message in the provider payload); and
        - a *rolling* one on the final message, so the conversation prefix
          that has grown since the previous iteration -- prior assistant
          turns and (often large) tool results -- is cached too, instead of
          being re-billed at the full input rate every ReAct iteration.

        Every message's flag is reset first so breakpoints never accumulate
        past the small per-request limit imposed by caching providers (e.g.
        4 on Anthropic via OpenRouter). Providers without prompt caching
        ignore the markers, so this is a no-op for them.
        """
        if not messages:
            return
        for msg in messages:
            msg.cache = False
        if messages[0].role == "system":
            messages[0].cache = True
        messages[-1].cache = True

    def _trim_history(self, messages: list[Message]) -> None:
        """Drop oldest ReAct iterations beyond ``history_window``.

        An "iteration" starts at an assistant message that carries
        ``tool_calls`` and extends through its matching ``role="tool"``
        replies. Whole iteration groups are dropped together so every
        ``tool_call_id`` keeps its paired tool-result message (required by
        OpenAI-compatible APIs). The system message and the initial user
        query are never touched.
        """
        if not self.history_window or self.history_window <= 0:
            return

        iter_starts = [
            i for i, m in enumerate(messages)
            if m.role == "assistant" and m.tool_calls
        ]
        if len(iter_starts) <= self.history_window:
            return

        first_keep = iter_starts[len(iter_starts) - self.history_window]
        first_drop = iter_starts[0]
        dropped = first_keep - first_drop
        del messages[first_drop:first_keep]
        logger.debug(
            f"Trimmed {dropped} message(s) from history "
            f"(window={self.history_window}, iterations kept={self.history_window})"
        )

    async def _get_response(
        self, messages: list[Message], run: AgentRun | None = None
    ) -> ProviderResponse:
        """Call the provider, recovering from context-window overflow by pruning.

        On ``ContextWindowExceededError`` the oldest ReAct iteration is dropped
        and the call is retried (no estimate involved -- we react to the
        provider's actual rejection). The loop ends when the prompt fits or
        there is nothing left to prune, in which case the error propagates.
        """
        while True:
            self._apply_cache_breakpoints(messages)
            try:
                return await self._retry_complete(
                    messages=messages,
                    tools=self.tools if self.tools else None,
                )
            except ContextWindowExceededError:
                if not self._prune_oldest_iteration(messages):
                    logger.error(
                        "Context window exceeded and no prunable history remains; "
                        "giving up (a single tool result may be too large)."
                    )
                    raise
                if run is not None:
                    run.context_prunes += 1
                logger.warning(
                    "Context window exceeded; pruned oldest iteration and retrying."
                )

    def _prune_oldest_iteration(self, messages: list[Message]) -> bool:
        """Drop the oldest assistant+tool_result group to shrink the prompt.

        Preserves the system message, the initial user query, the most recent
        iteration, and every ``tool_call_id`` pairing. Returns True if a group
        was dropped, False if there is nothing safe left to drop.
        """
        iter_starts = [
            i for i, m in enumerate(messages)
            if m.role == "assistant" and m.tool_calls
        ]
        # Need at least two iteration groups: drop the oldest, keep a newer one.
        if len(iter_starts) < 2:
            return False
        first_drop, first_keep = iter_starts[0], iter_starts[1]
        dropped = first_keep - first_drop
        del messages[first_drop:first_keep]
        logger.debug(
            f"Context-overflow recovery: pruned {dropped} message(s) from the "
            "oldest iteration."
        )
        return True

    async def _retry_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> ProviderResponse:
        total_attempts = 1 + self.max_retries

        async def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"Provider call failed (attempt {attempt + 1}/{total_attempts}), "
                f"retrying in {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)

        try:
            return await retry_with_backoff(
                lambda: self.provider.complete(messages, tools=tools, temperature=0.0),
                max_retries=self.max_retries,
                base_delay=self.retry_base_delay,
                on_retry=on_retry,
                # A context-window overflow is deterministic -- don't waste retries
                # on it; let it surface for prune-and-retry in _get_response.
                retryable=lambda exc: not isinstance(exc, ContextWindowExceededError),
            )
        except ContextWindowExceededError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.provider.provider_name) from exc

    async def _process_tool_calls(
        self,
        response: ProviderResponse,
        messages: list[Message],
        run: AgentRun,
        iteration: int = 0,
    ) -> bool:
        """Process tool calls from the provider response.

        Returns:
            True if the agent loop should continue, False if a non-recoverable
            tool error requires stopping immediately.
        """
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "thought_signature": tc.thought_signature,
                    }
                    for tc in response.tool_calls
                ]
                if response.tool_calls
                else None,
            )
        )

        # Log the model's reasoning as a THOUGHT step (owns the LLM latency + tokens).
        run.steps.append(
            AgentStep(
                step_type=StepType.THOUGHT,
                content=response.content or "",
                iteration=iteration,
                tokens_used=response.input_tokens + response.output_tokens,
                latency_ms=response.latency_ms,
                reasoning=response.reasoning_content,
            )
        )

        llm_call_timestamp = time.time()

        for tool_call in response.tool_calls or []:
            run.tool_calls_count += 1

            action_step = AgentStep(
                step_type=StepType.ACTION,
                content=f"Calling {tool_call.name}",
                iteration=iteration,
                tool_name=tool_call.name,
                tool_input=tool_call.arguments,
                # Latency belongs to the THOUGHT step (the LLM call); tool
                # execution latency is captured on the OBSERVATION step.
                latency_ms=0.0,
                timestamp=llm_call_timestamp,
            )
            run.steps.append(action_step)

            logger.debug(f"Executing tool: {tool_call.name} with args: {json.dumps(tool_call.arguments, indent=2)}")

            start_time = time.time()
            try:
                tool_result = await self._execute_tool(
                    tool_call.name, tool_call.arguments
                )
            except TimeoutError:
                error_payload = {"error": f"Tool '{tool_call.name}' timed out after {self.tool_timeout}s"}
                logger.error(error_payload["error"])
                tool_result = json.dumps(error_payload)
            except Exception as e:
                error_payload = {"error": str(e), "tool": tool_call.name, "error_type": type(e).__name__}
                logger.error(f"Tool '{tool_call.name}' failed: {e}", exc_info=True)
                tool_result = json.dumps(error_payload)

            execution_time = (time.time() - start_time) * 1000
            self._log_tool_output(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                iteration=iteration,
                execution_time_ms=execution_time,
                tool_result=tool_result,
            )

            # Check for non-recoverable tool failure before processing further.
            try:
                result_data = json.loads(tool_result)
                if (
                    isinstance(result_data, dict)
                    and not result_data.get("success", True)
                    and result_data.get("metadata", {}).get("recoverable") is False
                ):
                    error_msg = result_data.get("error") or f"Non-recoverable error in tool '{tool_call.name}'"
                    logger.error(f"Non-recoverable tool error: {error_msg}")
                    run.success = False
                    run.error = error_msg
                    run.steps.append(AgentStep(step_type=StepType.ERROR, content=error_msg))
                    return False
            except json.JSONDecodeError:
                pass  # Not valid JSON or unexpected shape — treat as recoverable

            context_result, csv_path = self._result_processor.process_result(
                tool_call.name, tool_result
            )

            if self.max_tool_result_chars > 0 and len(context_result) > self.max_tool_result_chars:
                logger.warning(
                    f"Tool {tool_call.name} result truncated from {len(context_result)} "
                    f"to {self.max_tool_result_chars} chars before adding to context"
                )
                context_result = (
                    context_result[: self.max_tool_result_chars]
                    + f"\n...[truncated: result exceeded {self.max_tool_result_chars} chars]"
                )

            logger.debug(f"Tool {tool_call.name} returned {len(tool_result)} chars")
            if csv_path:
                logger.info(f"Large result saved to CSV: {csv_path}")

            obs_step = AgentStep(
                step_type=StepType.OBSERVATION,
                content=context_result,
                iteration=iteration,
                tool_name=tool_call.name,
                tool_output=tool_result,
                latency_ms=execution_time,
            )
            run.steps.append(obs_step)

            tool_message = self._create_tool_message(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=tool_result,
                context_result=context_result,
            )
            messages.append(tool_message)

        self._trim_history(messages)

        return True

    def _log_tool_output(
        self,
        tool_name: str,
        tool_call_id: str,
        iteration: int,
        execution_time_ms: float,
        tool_result: str,
    ) -> None:
        is_json, parsed_output = self._parse_tool_output_json(tool_result)
        is_error = self._is_tool_output_error(tool_result, parsed_output, is_json)
        output_chars = len(tool_result)

        logger.info(
            "Tool output metadata | "
            f"tool={tool_name} latency_ms={execution_time_ms:.1f} "
            f"output_chars={output_chars} is_json={is_json} "
            f"is_error={is_error} mode={self.tool_output_log_mode}"
        )

        if self.tool_output_log_mode == "off":
            return

        if self.tool_output_log_mode == "errors_only":
            if not is_error:
                return
            preview = self._build_tool_output_preview(tool_result)
            logger.error(f"Tool error output preview ({tool_name}):\n{preview}")
            return

        if self.tool_output_log_mode == "preview":
            preview = self._build_tool_output_preview(tool_result)
            logger.info(f"Tool output preview ({tool_name}):\n{preview}")
            return

        if self.tool_output_log_mode == "full":
            file_path = self._write_full_tool_output(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                iteration=iteration,
                tool_result=tool_result,
            )
            if file_path:
                logger.info(f"Tool output saved to file: {file_path}")
            return

        logger.warning(
            f"Unknown tool_output_log_mode={self.tool_output_log_mode!r}. "
            "Supported modes: off, errors_only, preview, full."
        )

    def _parse_tool_output_json(self, tool_result: str) -> tuple[bool, Any | None]:
        try:
            return True, json.loads(tool_result)
        except json.JSONDecodeError:
            return False, None

    def _is_tool_output_error(
        self,
        tool_result: str,
        parsed_output: Any | None,
        is_json: bool,
    ) -> bool:
        if is_json and isinstance(parsed_output, dict):
            if self._dict_indicates_error(parsed_output):
                return True

            nested_data = parsed_output.get("data")
            if isinstance(nested_data, str):
                try:
                    nested_output = json.loads(nested_data)
                except json.JSONDecodeError:
                    nested_output = None
                if isinstance(nested_output, dict) and self._dict_indicates_error(nested_output):
                    return True
            return False

        lowered = tool_result.lower()
        return any(token in lowered for token in ("error", "exception", "traceback"))

    @staticmethod
    def _dict_indicates_error(payload: dict[str, Any]) -> bool:
        error_value = payload.get("error")
        if error_value not in (None, "", False):
            return True
        success = payload.get("success")
        if success is False:
            return True
        status = payload.get("status")
        return isinstance(status, str) and status.lower() == "error"

    def _build_tool_output_preview(self, tool_result: str) -> str:
        preview = self._redact_tool_output(tool_result)
        if self.tool_output_log_max_chars <= 0:
            return "[preview omitted: tool_output_log_max_chars=0]"
        if len(preview) <= self.tool_output_log_max_chars:
            return preview
        return (
            preview[: self.tool_output_log_max_chars]
            + f"\n...[preview truncated at {self.tool_output_log_max_chars} chars]"
        )

    def _redact_tool_output(self, tool_result: str) -> str:
        if not self.tool_output_redact_secrets:
            return tool_result

        redacted = _SECRET_JSON_VALUE_RE.sub(r"\1[REDACTED]\3", tool_result)
        redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", redacted)
        redacted = _BEARER_TOKEN_RE.sub(r"\1 [REDACTED]", redacted)
        return redacted

    def _write_full_tool_output(
        self,
        tool_name: str,
        tool_call_id: str,
        iteration: int,
        tool_result: str,
    ) -> Path | None:
        try:
            self.tool_output_log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            safe_tool = self._sanitize_path_component(tool_name)
            safe_call = self._sanitize_path_component(tool_call_id)
            file_name = f"iter_{iteration + 1}_{safe_tool}_{safe_call}_{timestamp}.log"
            output_path = self.tool_output_log_dir / file_name
            output_path.write_text(self._redact_tool_output(tool_result), encoding="utf-8")
            return output_path
        except Exception as exc:
            logger.warning(f"Failed to write tool output log file for {tool_name}: {exc}")
            return None

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
        return sanitized or "unknown"

    def _create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        context_result: str,
    ) -> Message:
        """Create a tool message, handling images from RAG results."""
        images = self._result_processor.extract_images(result)

        if not images:
            return Message(
                role="tool",
                content=context_result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        content_parts: list[TextContent | ImageContent] = [TextContent(text=context_result)]

        for img in images:
            content_parts.append(
                ImageContent(
                    image_base64=img.get("base64", img.get("image_base64", "")),
                    media_type=img.get("media_type", "image/jpeg"),
                )
            )

        logger.debug(f"Tool {tool_name} returned {len(images)} image(s)")

        return Message(
            role="tool",
            content=context_result,
            content_parts=content_parts,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        if tool_name not in self._tool_registry:
            raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

        if inspect.iscoroutinefunction(self.tool_executor):
            result = await asyncio.wait_for(
                self.tool_executor(tool_name, arguments),
                timeout=self.tool_timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, self.tool_executor, tool_name, arguments),
                    timeout=self.tool_timeout,
                )
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=self.tool_timeout)

        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _default_tool_executor(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Default tool executor that returns an error."""
        return json.dumps(
            {
                "error": "No tool executor configured",
                "tool": tool_name,
                "arguments": arguments,
            }
        )
