"""
MCP Tool Registry — wraps MCP server tools as LangChain Tool objects.

Each MCP tool is exposed as a standard langchain_core.tools.Tool so it plugs
into the existing ResearchAgent without changing the LangGraph graph structure.

The registry auto-detects which servers are available (mcp package + uvx installed)
and returns only tools that can actually run in the current environment.
"""
from __future__ import annotations
import logging
import os
from typing import List, Optional

from langchain_core.tools import Tool

from src.mcp.servers import MCPServerDef, get_enabled_servers, get_server

logger = logging.getLogger(__name__)


# ── Individual tool implementations ───────────────────────────────────────────

def _make_fetch_tool(server: MCPServerDef) -> Tool:
    """mcp-server-fetch: fetches full content of a URL."""
    from src.mcp.client import call_mcp_tool_sync

    def _fetch(url: str) -> str:
        result = call_mcp_tool_sync(
            server_command=server.command,
            tool_name="fetch",
            arguments={"url": url, "max_length": 8000},
        )
        if result:
            return result[:8000]
        return f"Could not fetch content from: {url}"

    return Tool(
        name="mcp_fetch",
        description=(
            "Fetch the full text content of a web page by URL. "
            "Use this to get the complete article text when a URL is known. "
            "Input: a single URL string."
        ),
        func=_fetch,
    )


def _make_brave_search_tool(server: MCPServerDef) -> Tool:
    """mcp-server-brave-search: higher-quality web search."""
    from src.mcp.client import call_mcp_tool_sync
    brave_key = os.environ.get("BRAVE_API_KEY", "")

    def _brave_search(query: str) -> str:
        result = call_mcp_tool_sync(
            server_command=server.command,
            tool_name="brave_web_search",
            arguments={"query": query, "count": 5},
            env={"BRAVE_API_KEY": brave_key},
        )
        if result:
            return result
        return f"Brave Search returned no results for: {query}"

    return Tool(
        name="mcp_brave_search",
        description=(
            "Search the web using Brave Search API. Higher quality results than DuckDuckGo. "
            "Input: a search query string."
        ),
        func=_brave_search,
    )


def _make_arxiv_tool(server: MCPServerDef) -> Tool:
    """mcp-server-arxiv: academic paper search."""
    from src.mcp.client import call_mcp_tool_sync

    def _arxiv_search(query: str) -> str:
        result = call_mcp_tool_sync(
            server_command=server.command,
            tool_name="search_arxiv",
            arguments={"query": query, "max_results": 5},
        )
        if result:
            return result
        return f"arXiv returned no results for: {query}"

    return Tool(
        name="mcp_arxiv_search",
        description=(
            "Search arXiv for academic papers. Best for ML, AI, physics, math, "
            "and computer science topics. Returns paper titles, abstracts, and arXiv URLs. "
            "Input: a search query string."
        ),
        func=_arxiv_search,
    )


# ── Builder dispatch ───────────────────────────────────────────────────────────

_BUILDERS = {
    "fetch":        _make_fetch_tool,
    "brave-search": _make_brave_search_tool,
    "arxiv":        _make_arxiv_tool,
}


def get_mcp_tools(enabled_only: bool = True) -> List[Tool]:
    """
    Return all available MCP tools as LangChain Tool objects.
    If the mcp package is not installed, returns an empty list.
    """
    try:
        import mcp  # noqa: F401
    except ImportError:
        logger.info("[MCP] mcp package not installed — MCP tools unavailable")
        return []

    servers = get_enabled_servers() if enabled_only else [
        get_server(s) for s in _BUILDERS if get_server(s)
    ]

    tools = []
    for server in servers:
        if server is None:
            continue
        builder = _BUILDERS.get(server.id)
        if builder:
            try:
                tool = builder(server)
                tools.append(tool)
                logger.info("[MCP] Registered tool from server: %s", server.id)
            except Exception as exc:
                logger.warning("[MCP] Failed to build tool for %s: %s", server.id, exc)

    return tools


def get_mcp_tool_names() -> List[str]:
    """Return names of available MCP tools for UI display and state tracking."""
    return [t.name for t in get_mcp_tools()]


def is_mcp_available() -> bool:
    """Check if the mcp package and at least one server is available."""
    try:
        import mcp  # noqa: F401
        return len(get_enabled_servers()) > 0
    except ImportError:
        return False
