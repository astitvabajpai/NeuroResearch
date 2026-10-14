"""
MCP Server Registry — defines available MCP servers and their tools.

On HuggingFace Spaces, uvx subprocess spawning is blocked, so all MCP
servers are automatically disabled and tools fall back to the built-in
HTTP fetcher.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Detect HF Spaces environment
_ON_HF_SPACES = os.environ.get("SPACE_ID") is not None


@dataclass
class MCPServerDef:
    id:                  str
    name:                str
    command:             List[str]
    tools:               List[str]
    description:         str
    requires_env:        List[str] = field(default_factory=list)
    enabled_by_default:  bool      = True


MCP_SERVERS: List[MCPServerDef] = [
    MCPServerDef(
        id          = "fetch",
        name        = "MCP Fetch",
        command     = ["uvx", "mcp-server-fetch"],
        tools       = ["fetch"],
        description = "Fetches the full content of any URL. Used to get complete article "
                      "text instead of DuckDuckGo snippets.",
        requires_env= [],
        enabled_by_default=True,
    ),
    MCPServerDef(
        id          = "brave-search",
        name        = "Brave Search",
        command     = ["uvx", "mcp-server-brave-search"],
        tools       = ["brave_web_search", "brave_local_search"],
        description = "Higher-quality web search via Brave Search API. "
                      "Requires BRAVE_API_KEY environment variable.",
        requires_env= ["BRAVE_API_KEY"],
        enabled_by_default=True,   # enabled when BRAVE_API_KEY is set
    ),
    MCPServerDef(
        id          = "arxiv",
        name        = "arXiv Search",
        command     = ["uvx", "mcp-server-arxiv"],
        tools       = ["search_arxiv", "get_paper"],
        description = "Search arXiv for academic papers. Best for ML, physics, "
                      "math, and computer science topics.",
        requires_env= [],
        enabled_by_default=False,   # off by default — not all topics need academic papers
    ),
]

# Index by id for O(1) lookup
_SERVER_BY_ID = {s.id: s for s in MCP_SERVERS}


def get_server(server_id: str) -> Optional[MCPServerDef]:
    return _SERVER_BY_ID.get(server_id)


def get_enabled_servers(env: Optional[dict] = None) -> List[MCPServerDef]:
    """
    Return servers that are enabled by default AND have all required env vars.
    Returns empty list on HuggingFace Spaces (no subprocess support).
    """
    if _ON_HF_SPACES:
        return []
    import os as _os
    check_env = env or dict(_os.environ)
    result = []
    for s in MCP_SERVERS:
        if not s.enabled_by_default:
            continue
        missing = [k for k in s.requires_env if not check_env.get(k)]
        if missing:
            continue
        result.append(s)
    return result


def list_tool_names() -> List[str]:
    """All tool names across all defined MCP servers."""
    tools = []
    for s in MCP_SERVERS:
        tools.extend(s.tools)
    return tools
