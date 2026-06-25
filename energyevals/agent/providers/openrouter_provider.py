import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from energyevals.agent.constants import MAX_TOKENS
from energyevals.agent.exceptions import ContextWindowExceededError, ProviderError
from energyevals.agent.schema.messages import ImageContent, TextContent

from energyevals.agent.providers.base_provider import (
    BaseProvider,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)

_LLAMA_TOOL_CALL_RE = re.compile(
    r"<function=(\w+)(.*?)</function>",
    re.DOTALL,
)

_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Substrings that mark a context-window overflow, regardless of whether it
# arrives as a pre-flight 400 or as an upstream 502 error body. Providers phrase
# it differently -- OpenAI "maximum context length", OpenRouter "exceeds the
# context window", Moonshot/Kimi "exceeded model token limit" -- so match all.
_CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "exceeds the context window",
    "exceeded model token limit",
    "exceeds the maximum number of tokens",
)


def _is_context_error(message: str | None) -> bool:
    """True if an error message indicates the prompt exceeded the context window."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_ERROR_MARKERS)


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider implementation.

    OpenRouter (https://openrouter.ai) exposes hundreds of models from many
    vendors behind a single OpenAI-compatible ``/chat/completions`` endpoint.

    Reasoning tokens:
        OpenRouter accepts a non-standard ``reasoning`` request parameter that
        controls model "thinking" (see
        https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
        This provider forwards ``{"reasoning": {"effort": <effort>}}`` via the
        OpenAI SDK ``extra_body`` when an ``effort`` is configured.

        ``effort`` is a free-form string (``none``, ``minimal``, ``low``,
        ``medium``, ``high``, ``xhigh`` -- the accepted set is model-dependent).
        It is forwarded verbatim; OpenRouter normalises effort to a token
        budget per model and rejects unsupported values with an API error.

        When the model returns a reasoning trace it is captured into
        ``ProviderResponse.reasoning_content``; the token count is captured
        into ``ProviderResponse.reasoning_tokens``.

    Provider routing:
        OpenRouter routes each model slug to one of several upstream inference
        providers, and prompt caching only happens when the chosen upstream
        supports it. Pass ``provider_routing`` (OpenRouter's ``provider``
        object, e.g. ``{"only": ["wandb"]}``) to pin a caching-capable
        upstream instead of relying on OpenRouter's default load balancing.
        See https://openrouter.ai/docs/features/provider-routing.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        effort: str | None = None,
        provider_routing: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenRouter provider.

        Args:
            model: OpenRouter model slug (e.g. ``"openai/gpt-5-mini"``,
                ``"anthropic/claude-sonnet-4.5"``, ``"deepseek/deepseek-v3.2"``).
            api_key: OpenRouter API key. Defaults to ``OPENROUTER_API_KEY`` env var.
            base_url: Optional base URL override.
            effort: Reasoning effort forwarded as ``reasoning.effort`` on every
                request. Free-form string; the accepted set is model-dependent
                (commonly ``none``/``minimal``/``low``/``medium``/``high``/``xhigh``).
                ``None`` leaves reasoning at the model/provider default.
            provider_routing: OpenRouter provider-routing object, forwarded
                verbatim as the request ``provider`` field on every call (e.g.
                ``{"only": ["wandb"]}`` to pin the upstream inference provider).
                ``None`` uses OpenRouter's default routing.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = base_url or self.OPENROUTER_BASE_URL

        super().__init__(model, api_key, base_url, **kwargs)

        self.effort = effort
        self.provider_routing = provider_routing

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        del self.api_key

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _build_extra_body(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the ``extra_body`` payload for non-standard OpenRouter params.

        Carries the ``reasoning`` parameter (derived from ``effort``) and the
        ``provider`` routing object. A per-call ``effort`` / ``provider_routing``
        in ``kwargs`` overrides the corresponding instance default.
        """
        extra_body: dict[str, Any] = {}

        effort = kwargs.get("effort", self.effort)
        if effort:
            extra_body["reasoning"] = {"effort": effort}

        provider_routing = kwargs.get("provider_routing", self.provider_routing)
        if provider_routing:
            provider = dict(provider_routing)          
            provider.setdefault("allow_fallbacks", False)  
            extra_body["provider"] = provider

        return extra_body

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion using OpenRouter's API."""
        start_time = time.time()

        formatted_messages = self.format_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        extra_body = self._build_extra_body(kwargs)
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        # ``effort``/``reasoning_effort`` and ``provider_routing`` are folded
        # into ``extra_body`` above; ``tool_choice`` is handled separately.
        # Don't forward any of them as raw top-level params.
        excluded = {"tool_choice", "reasoning_effort", "effort", "provider_routing"}
        request_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in request_kwargs and k not in excluded}
        )

        try:
            response = await self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            # The pre-flight 400 ("maximum context length is N tokens") arrives
            # as a raised client error; surface it as a typed overflow.
            if _is_context_error(str(exc)):
                raise ContextWindowExceededError(
                    str(exc), provider=self.provider_name, model=self.model
                ) from exc
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Some upstreams return 200 with NO choices and an ``error`` body (e.g. a
        # 502 "input exceeds the context window"). Only inspect ``error`` when
        # there are no usable choices -- a normal response has no ``error`` and we
        # must not let it shadow the happy path. Gating on choices also avoids
        # misfiring on test doubles whose attributes are all truthy.
        api_error = None if getattr(response, "choices", None) else getattr(response, "error", None)
        if api_error is not None:
            err_msg = (
                api_error.get("message")
                if isinstance(api_error, dict)
                else str(api_error)
            )
            if _is_context_error(err_msg):
                raise ContextWindowExceededError(
                    err_msg or str(api_error),
                    provider=self.provider_name,
                    model=self.model,
                )
            raise ProviderError(
                f"API returned an error: {err_msg or api_error}",
                provider=self.provider_name,
                model=self.model,
            )

        try:
            tool_calls = None
            message = response.choices[0].message
            content = message.content or ""
            reasoning_content = self._extract_reasoning_content(message)

            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        parsed_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=parsed_args)
                    )

            if not tool_calls and content:
                tool_calls = self._parse_text_tool_calls(content)
                if tool_calls:
                    logger.debug(
                        f"Parsed {len(tool_calls)} tool call(s) from text content "
                        f"(model returned no native tool_calls field)"
                    )
                    content = _LLAMA_TOOL_CALL_RE.sub("", content)
                    content = _QWEN_TOOL_CALL_RE.sub("", content).strip()

            cached_tokens = 0
            reasoning_tokens = 0
            if response.usage:
                prompt_details = getattr(response.usage, "prompt_tokens_details", None)
                if prompt_details is not None:
                    cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                completion_details = getattr(
                    response.usage, "completion_tokens_details", None
                )
                if completion_details is not None:
                    reasoning_tokens = (
                        getattr(completion_details, "reasoning_tokens", 0) or 0
                    )

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                cached_tokens=cached_tokens,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                reasoning_tokens=reasoning_tokens,
                reasoning_content=reasoning_content,
                latency_ms=latency_ms,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )
        except (KeyError, AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Malformed API response: {exc}. Raw: {response!r}",
                provider=self.provider_name,
            ) from exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenRouter."""
        formatted_messages = self.format_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            request_kwargs["tools"] = self.format_tools(tools)
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        extra_body = self._build_extra_body(kwargs)
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _extract_reasoning_content(message: Any) -> str | None:
        """Extract the reasoning trace text from an OpenRouter response message.

        OpenRouter surfaces the trace as ``message.reasoning`` (a plain string)
        and/or ``message.reasoning_details`` (a list of structured blocks). The
        ``reasoning`` string is preferred; the details list is joined as a
        fallback for providers that only populate the structured form.
        """
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        details = getattr(message, "reasoning_details", None)
        if isinstance(details, list):
            texts: list[str] = []
            for block in details:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("summary") or ""
                else:
                    text = getattr(block, "text", "") or getattr(block, "summary", "")
                if text:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
        return None

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for OpenAI-compatible function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for OpenRouter's API (OpenAI-compatible).

        When ``msg.cache`` is set, the content is emitted as a structured
        content list carrying a ``cache_control: ephemeral`` marker. OpenRouter
        forwards this to providers that support prompt caching (Anthropic,
        Gemini, DeepSeek) so the system prompt + tool schemas can be cached
        across iterations of the ReAct loop and across benchmark questions.
        Providers without explicit cache control (e.g. OpenAI, which caches
        automatically) ignore the marker.
        """
        formatted = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {
                "role": msg.role,
            }
            if msg.content_parts and msg.has_images:
                formatted_msg["content"] = self._format_multimodal_content(msg)
            elif msg.cache and msg.content:
                formatted_msg["content"] = [
                    {
                        "type": "text",
                        "text": msg.content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                formatted_msg["content"] = msg.content

            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool" and msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id

            formatted.append(formatted_msg)

        return formatted

    def _format_multimodal_content(self, msg: Message) -> list[dict[str, Any]]:
        """Format multi-modal content (text + images) for OpenAI-compatible API."""
        content_parts: list[dict[str, Any]] = []
        for part in self._extract_content_parts(msg.content_parts or []):
            if isinstance(part, TextContent):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.image_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url},
                    })
                elif part.image_base64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part.media_type};base64,{part.image_base64}"
                        },
                    })
        return content_parts

    @staticmethod
    def _parse_text_tool_calls(content: str) -> list[ToolCall] | None:
        """Parse tool calls emitted as raw text instead of structured fields.

        Supports two conventions observed in the wild:

        - Qwen / Hermes: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
        - Llama 3: ``<function=name{"arg": ...}</function>``
        """
        tool_calls: list[ToolCall] = []

        for raw in _QWEN_TOOL_CALL_RE.findall(content):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed Qwen-style tool_call payload: {raw[:500]!r}")
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(arguments, dict):
                continue
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=name,
                    arguments=arguments,
                )
            )

        for func_name, raw_args in _LLAMA_TOOL_CALL_RE.findall(content):
            raw_args = raw_args.strip().rstrip(">")
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning(
                    f"Skipping malformed Llama-style arguments for '{func_name}': "
                    f"{raw_args[:500]!r}"
                )
                continue
            if not isinstance(arguments, dict):
                continue
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=func_name,
                    arguments=arguments,
                )
            )

        return tool_calls or None
