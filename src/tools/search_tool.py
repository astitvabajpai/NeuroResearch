"""
Search tool — Tavily primary, DuckDuckGo fallback.

Priority:
  1. Tavily (if TAVILY_API_KEY is set) — returns full content per result,
     optimised for LLM pipelines, includes clean URLs and raw_content.
  2. DuckDuckGo (ddgs) — free, no key, retries with exponential backoff.
  3. DuckDuckGo Instant Answer API — final fallback if ddgs rate-limits.
"""
from __future__ import annotations
import logging
import time
import random
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


# ── Tavily ─────────────────────────────────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 5) -> str:
    """Search using Tavily API. Returns rich results with full content."""
    try:
        from src.config.settings import get_settings
        api_key = get_settings().TAVILY_API_KEY
        if not api_key:
            return ""
    except Exception:
        return ""

    try:
        from tavily import TavilyClient
        client   = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,   # full page content, not just snippets
            search_depth="advanced",    # deeper crawl for better results
        )
        results  = response.get("results", [])
        if not results:
            return ""

        parts = []
        for r in results:
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = r.get("content", "")
            # Cap raw_content tightly — it's rich but eats tokens fast
            raw     = r.get("raw_content") or ""
            body    = (raw[:1500] if raw else content[:800]).strip()
            parts.append(f"**{title}**\n{body}\nSource: {url}")

        logger.info("[Search] Tavily returned %d results for: %s", len(results), query[:60])
        return "\n\n".join(parts)

    except ImportError:
        logger.warning("[Search] tavily-python not installed — run: pip install tavily-python")
        return ""
    except Exception as exc:
        logger.warning("[Search] Tavily error: %s", exc)
        return ""


# ── DuckDuckGo ─────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5, retries: int = 4) -> str:
    """Search using DuckDuckGo with retry on rate limits."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return _ddg_instant_answer(query)

    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for: {query}"
            parts = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                parts.append(f"**{title}**\n{body}\nSource: {href}")
            return "\n\n".join(parts)
        except Exception as e:
            err = str(e)
            if "202" in err or "ratelimit" in err.lower() or "rate" in err.lower():
                wait = (2 ** attempt) + random.uniform(1, 3)
                logger.warning("[Search] DDG rate limited (attempt %d/%d), waiting %.1fs",
                               attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            logger.warning("[Search] DDG error: %s", e)
            return _ddg_instant_answer(query)

    logger.warning("[Search] DDG all retries exhausted, using instant answer fallback")
    return _ddg_instant_answer(query)


def _ddg_instant_answer(query: str) -> str:
    """DuckDuckGo Instant Answer API — different endpoint, less rate-limited."""
    try:
        import urllib.request
        import urllib.parse
        import json
        params = urllib.parse.urlencode({
            "q": query, "format": "json",
            "no_html": "1", "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        parts = []
        if data.get("AbstractText"):
            parts.append(f"**{data.get('Heading', query)}**\n{data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        if parts:
            return "\n\n".join(parts)
    except Exception as e:
        logger.warning("[Search] DDG instant answer failed: %s", e)

    return (
        f"Search unavailable for '{query}'. "
        "Please generate a research response based on your training knowledge."
    )


# ── Unified search ─────────────────────────────────────────────────────────────

def _search(query: str, max_results: int = 5) -> str:
    """
    Try Tavily first (if key is set), fall back to DuckDuckGo.
    Logs which provider was used.
    """
    result = _tavily_search(query, max_results=max_results)
    if result:
        return result
    logger.info("[Search] Falling back to DuckDuckGo for: %s", query[:60])
    return _ddg_search(query, max_results=max_results)


def get_search_tool() -> Tool:
    """Return the unified search tool (Tavily → DuckDuckGo)."""
    # Log which provider will be used at tool creation time
    try:
        from src.config.settings import get_settings
        key = get_settings().TAVILY_API_KEY
        provider = "Tavily" if key else "DuckDuckGo"
    except Exception:
        provider = "DuckDuckGo"
    logger.info("[Search] Active provider: %s", provider)

    return Tool(
        name="web_search",
        description=(
            "Search the web for information about a topic. "
            f"Currently using: {provider}."
        ),
        func=_search,
    )


def get_search_provider() -> str:
    """Return the name of the active search provider."""
    try:
        from src.config.settings import get_settings
        return "Tavily" if get_settings().TAVILY_API_KEY else "DuckDuckGo"
    except Exception:
        return "DuckDuckGo"
