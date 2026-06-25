import json
import os
from typing import Any, Literal

from exa_py import Exa
from loguru import logger

from energyevals.utils import call_with_retry

from energyevals.tools.base_tool import BaseTool, tool_method
from energyevals.tools.constants import (
    SEARCH_HIGHLIGHTS_MAX_CHARS,
    SEARCH_MAX_RESULTS,
    SEARCH_TEXT_MAX_CHARS,
)

# The exa_py SDK raises plain exceptions (no Response); detect rate limits by message.
_EXA_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests")


def _is_exa_rate_limit(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _EXA_RATE_LIMIT_MARKERS)


def _normalize_flag(value: object) -> object:
    """Coerce a stringified boolean to a real bool; pass bool/dict/None through.

    Exa's text/highlights/summary options accept ``bool | dict``, but some models
    send the literal string ``"true"``/``"false"`` -- which Exa rejects, and where
    ``"false"`` is truthy so it would wrongly request the content. Normalize here.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return value


class SearchTool(BaseTool):

    def __init__(
        self,
        api_key: str | None = None,
        text_length_limit: int | None = None,
        default_num_results: int = SEARCH_MAX_RESULTS,
        max_num_results: int = SEARCH_MAX_RESULTS,
        text_max_chars: int = SEARCH_TEXT_MAX_CHARS,
        highlights_max_chars: int = SEARCH_HIGHLIGHTS_MAX_CHARS,
    ):
        super().__init__(
            name="search",
            description="Web search using Exa for finding energy-related information",
        )
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.text_length_limit = text_length_limit
        self.default_num_results = default_num_results
        self.max_num_results = max_num_results
        self.text_max_chars = text_max_chars
        self.highlights_max_chars = highlights_max_chars
        self._exa = Exa(self.api_key) if self.api_key else None

        if not self.api_key:
            logger.warning("EXA_API_KEY not set. Search functionality will be limited.")

    @staticmethod
    def _capped_content_option(
        value: bool | dict | None, cap: int
    ) -> dict | None:
        """Normalize a bool|dict content flag into a capped Exa options object.

        A falsy flag means "don't request this content" (returns None so it's
        dropped from the request). Otherwise a ``maxCharacters`` cap is enforced:
        the caller's value is honored only if it's smaller than ``cap``, so the
        model cannot bypass the limit by passing ``text=True`` or a larger cap.

        Also coerces shapes models commonly get wrong, which Exa rejects: a
        stringified boolean (``"true"``/``"false"`` instead of a real bool, where
        ``"false"`` is truthy and would wrongly request content), ``highlights.query``
        passed as a list of keywords (Exa requires a single string), and a snake_case
        ``max_characters`` key (folded into the cap).
        """
        value = _normalize_flag(value)
        if not value:
            return None
        option = dict(value) if isinstance(value, dict) else {}
        # Accept snake_case max_characters as the requested cap, then drop it so the
        # request carries only the canonical maxCharacters.
        requested = option.pop("max_characters", None)
        if requested is None:
            requested = option.get("maxCharacters")
        option["maxCharacters"] = (
            min(requested, cap) if isinstance(requested, int) and requested > 0 else cap
        )
        # Exa's highlights.query must be a single string; models sometimes pass a
        # list of keywords -> join into one query so the request validates.
        query = option.get("query")
        if isinstance(query, (list, tuple)):
            option["query"] = " ".join(str(q) for q in query if str(q).strip())
        return option

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Extract url, title, author, date, text, and highlights from an Exa result."""
        result_dict: dict[str, Any] = {"url": result.url}
        if result.title:
            result_dict["title"] = result.title
        if getattr(result, "author", None):
            result_dict["author"] = result.author
        if getattr(result, "published_date", None):
            result_dict["published_date"] = result.published_date
        if result.text:
            result_dict["text"] = (
                result.text[:self.text_length_limit]
                if self.text_length_limit is not None
                else result.text
            )
        if getattr(result, "highlights", None):
            result_dict["highlights"] = result.highlights
        return result_dict

    @tool_method(name="search_web")
    def search(
        self,
        query: str,
        num_results: int | None = None,
        text: bool = True,
        highlights: bool = True,
        summary: bool = False,
        livecrawl: Literal["never", "fallback", "preferred", "always"] = "fallback",
        search_type: Literal["neural", "fast", "auto", "deep"] = "auto",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        """Search the web using Exa for relevant information about energy markets, regulations, companies, and technical topics.
        Returns titles, URLs, and text snippets from matching pages.

        Args:
            query: The search query describing what you're looking for.
            num_results: Number of results to return.
            text: Include text content in results.
            highlights: Include highlights in results.
            summary: Include Exa summaries when available.
            livecrawl: Live crawl behavior ("never", "fallback", "preferred", "always").
            search_type: Search method. "neural" uses embeddings for semantic similarity, "fast" is
                keyword-based, "auto" (default) combines both methods, "deep" performs comprehensive
                search with query expansion.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.

        Returns:
            JSON string with search results including query, num_results, and a list of result
            objects each containing url, title, author, published_date, text, and highlights.
        """
        if not self._exa:
            return json.dumps({"error": "EXA_API_KEY not configured"})

        try:
            capped_num = max(1, min(
                num_results if num_results is not None else self.default_num_results,
                self.max_num_results,
            ))

            kwargs: dict[str, Any] = {
                "text": self._capped_content_option(text, self.text_max_chars),
                "highlights": self._capped_content_option(highlights, self.highlights_max_chars),
                "summary": _normalize_flag(summary),
                "num_results": capped_num,
                "livecrawl": livecrawl,
                "type": search_type,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None and v != "" and v != []}

            results = call_with_retry(
                lambda: self._exa.search_and_contents(query, **kwargs),
                is_retryable=_is_exa_rate_limit,
            )

            parsed = [self._parse_result(r) for r in results.results]
            return json.dumps(
                {"query": query, "num_results": len(parsed), "results": parsed},
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return json.dumps({"error": str(e), "query": query})

    @tool_method(name="get_page_contents")
    def get_contents(
        self,
        urls: list[str],
        text: bool | dict = True,
        highlights: bool | dict = True,
        summary: bool | dict = False,
        livecrawl: Literal["never", "fallback", "preferred", "always"] = "fallback",
        subpages: int | None = None,
        subpage_target: list[str] | None = None,
    ) -> str:
        """Get the full page contents, summaries, and metadata for a list of URLs.
        Returns instant results from cache, with automatic live crawling as fallback for uncached pages.
        Use this after search_web to get more details from promising results.

        Args:
            urls: Array of URLs to crawl.
            text: If true, returns full page text. Can also be an object with custom settings.
            highlights: Include text snippets identified as most relevant.
                Can be a boolean or an object with configuration.
            summary: Include page summaries. Can be a boolean or an object with configuration.
            livecrawl: Live crawl behavior: 'never' (cached only), 'fallback' (cache first,
                live if unavailable), 'preferred' (live first, cache fallback), 'always'.
            subpages: Number of subpages to crawl from the provided URLs.
            subpage_target: Keywords to target specific subpages (e.g., ['about', 'products']).

        Returns:
            JSON string with page contents.
        """
        if not self._exa:
            return json.dumps({"error": "EXA_API_KEY not configured"})

        try:
            params: dict[str, Any] = {
                "urls": urls or [],
                "text": self._capped_content_option(text, self.text_max_chars),
                "highlights": self._capped_content_option(highlights, self.highlights_max_chars),
                "summary": _normalize_flag(summary),
                "livecrawl": livecrawl,
                "subpages": subpages,
                # exa_py validates snake_case option names and camelCases them
                # itself -- passing "subpageTarget" here is rejected as invalid.
                "subpage_target": subpage_target,
            }
            params = {k: v for k, v in params.items() if v is not None and v != "" and v != []}

            results = call_with_retry(
                lambda: self._exa.get_contents(**params),
                is_retryable=_is_exa_rate_limit,
            )

            parsed = [self._parse_result(r) for r in results.results]
            return json.dumps(
                {"num_results": len(parsed), "contents": parsed},
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Content retrieval failed: {e}")
            return json.dumps({"error": str(e), "urls": urls})
