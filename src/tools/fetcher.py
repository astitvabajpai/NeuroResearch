"""
Document Fetcher — fetches full page content from URLs found in search results.

Used by the RAG pipeline to get complete article text instead of 2-line snippets.
Gracefully handles timeouts, bad URLs, encoding issues, and paywalled pages.
"""
from __future__ import annotations
import logging
import re
import urllib.parse
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Hard caps to stay within token budgets
_MAX_PAGE_CHARS  = 4_000   # per page (reduced from 8000 to avoid 413s)
_REQUEST_TIMEOUT = 8       # seconds
_MAX_URLS        = 4       # max pages to fetch per research run (reduced from 6)

# Sites that consistently block scrapers or return garbage — skip them
_BLOCKED_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "reddit.com", "tiktok.com",
}


def _is_fetchable(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        if domain in _BLOCKED_DOMAINS:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        return True
    except Exception:
        return False


def _clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace to get plain readable text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # Remove script, style, nav, header, footer, ads
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except ImportError:
        # bs4 not installed — fall back to basic regex strip
        text = re.sub(r"<[^>]+>", " ", html)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url(url: str) -> Optional[str]:
    """
    Fetch a URL and return clean plain text, or None on failure.
    Caps output at _MAX_PAGE_CHARS characters.
    """
    if not _is_fetchable(url):
        logger.debug("[Fetcher] Skipping unfetchable URL: %s", url)
        return None
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NeuroResearch/1.0; "
                    "+https://github.com/astitvabajpai/NeuroResearch)"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
            allow_redirects=True,
        )
        if not resp.ok:
            logger.debug("[Fetcher] HTTP %s for %s", resp.status_code, url)
            return None

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            logger.debug("[Fetcher] Skipping non-HTML content-type: %s", content_type)
            return None

        text = _clean_html(resp.text)
        if len(text) < 100:
            # Page likely returned a bot-block page or minimal content
            return None

        truncated = text[:_MAX_PAGE_CHARS]
        logger.info("[Fetcher] Fetched %d chars from %s", len(truncated), url)
        return truncated

    except requests.exceptions.Timeout:
        logger.debug("[Fetcher] Timeout fetching %s", url)
        return None
    except Exception as exc:
        logger.debug("[Fetcher] Error fetching %s: %s", url, exc)
        return None


def extract_urls_from_search(search_text: str) -> list[str]:
    """Pull all https:// URLs out of a DuckDuckGo search result string."""
    urls = re.findall(r"https?://[^\s\)\"'<>]+", search_text)
    seen, unique = set(), []
    for url in urls:
        url = url.rstrip(".,)")
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def fetch_pages_from_search(search_text: str) -> list[dict]:
    """
    Extract URLs from search results, fetch each page, return list of
    {"url": str, "content": str} dicts for fetchable pages.
    Stops after _MAX_URLS successful fetches.
    """
    urls = extract_urls_from_search(search_text)
    pages = []
    for url in urls:
        if len(pages) >= _MAX_URLS:
            break
        content = fetch_url(url)
        if content:
            pages.append({"url": url, "content": content})
    logger.info("[Fetcher] Fetched %d pages from %d URLs", len(pages), len(urls))
    return pages
