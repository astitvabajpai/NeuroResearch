"""
Web search tool using DuckDuckGo (ddgs package).
Retries with exponential backoff on rate limits, falls back to
the DuckDuckGo Instant Answer API if all retries are exhausted.
"""
import time
import random
from langchain_core.tools import Tool


def _ddg_search(query: str, max_results: int = 5, retries: int = 4) -> str:
    """Search the web using DuckDuckGo with retry on rate limits."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return _fallback_search(query)

    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for: {query}"
            formatted = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                formatted.append(f"**{title}**\n{body}\nSource: {href}")
            return "\n\n".join(formatted)

        except Exception as e:
            err = str(e)
            if "202" in err or "ratelimit" in err.lower() or "rate" in err.lower():
                wait = (2 ** attempt) + random.uniform(1, 3)
                print(f"[Search] Rate limited (attempt {attempt+1}/{retries}), waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            print(f"[Search] Error: {e}")
            return _fallback_search(query)

    print("[Search] All retries exhausted, using fallback.")
    return _fallback_search(query)


def _fallback_search(query: str) -> str:
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
        print(f"[Search] Fallback failed: {e}")

    return (
        f"Search unavailable for '{query}'. "
        f"Please generate a research response based on your training knowledge."
    )


def get_search_tool() -> Tool:
    return Tool(
        name="web_search",
        description="Search the web for information about a topic.",
        func=_ddg_search,
    )
