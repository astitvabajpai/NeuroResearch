"""
MCP (Model Context Protocol) client layer.

Supports two transports:
  • SSE   — connects to an external MCP server via HTTP/SSE
            (set MCP_SERVER_URLS=http://host/sse in .env)
  • stdio — spawns the bundled arxiv_server as a subprocess
            (set MCP_USE_BUNDLED_ARXIV=true in .env, no external server needed)

If neither is configured the module returns None and the ResearchAgent
falls back to the built-in DuckDuckGo tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── MCP Tool (per-call, transport-agnostic) ───────────────────────────────────

class MCPTool:
    """
    Wraps a single MCP tool discovered from a server.
    Calls are dispatched through the owning MCPClient which manages
    the correct transport.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        client: "MCPClient",
    ) -> None:
        self.name         = name
        self.description  = description
        self.input_schema = input_schema
        self._client      = client

    @property
    def primary_arg(self) -> str:
        props    = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        return required[0] if required else next(iter(props), "query")

    def __call__(self, query: str) -> str:
        return self._client.call_tool_sync(self.name, {self.primary_arg: query})


# ── Base MCPClient ────────────────────────────────────────────────────────────

class MCPClient:
    """Abstract base — subclassed by SSEMCPClient and StdioMCPClient."""

    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}

    def has_tools(self) -> bool:
        return bool(self._tools)

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)

    def get_langchain_tools(self) -> list:
        from langchain_core.tools import Tool
        return [
            Tool(name=t.name, description=t.description, func=t)
            for t in self._tools.values()
        ]

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def call_tool_sync(self, name: str, arguments: dict) -> str:
        raise NotImplementedError


# ── SSE transport ─────────────────────────────────────────────────────────────

class SSEMCPClient(MCPClient):
    """Connects to one or more MCP servers via HTTP/SSE transport."""

    def __init__(self, server_urls: list[str]) -> None:
        super().__init__()
        self._server_urls = [u.strip() for u in server_urls if u.strip()]
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def connect(self) -> None:
        if not self._server_urls:
            return
        try:
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._discover_all())
            logger.info("[MCP/SSE] Ready. Tools: %s", list(self._tools))
        except Exception as exc:
            logger.warning("[MCP/SSE] Connection failed (%s) — falling back.", exc)
            self._tools = {}

    def close(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def call_tool_sync(self, name: str, arguments: dict) -> str:
        if self._loop is None or self._loop.is_closed():
            return "(MCP/SSE not available)"
        tool = self._tools.get(name)
        if tool is None:
            return f"(Unknown MCP tool: {name})"
        return self._loop.run_until_complete(
            self._call_sse_tool(tool._server_url, name, arguments)  # type: ignore[attr-defined]
        )

    async def _discover_all(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError("pip install mcp")

        for url in self._server_urls:
            try:
                async with sse_client(url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tool_list = await session.list_tools()
                        for t in tool_list.tools:
                            schema = getattr(t, "inputSchema", {}) or {}
                            mcp_tool = MCPTool(
                                name=t.name,
                                description=t.description or t.name,
                                input_schema=schema,
                                client=self,
                            )
                            mcp_tool._server_url = url  # type: ignore[attr-defined]
                            self._tools[t.name] = mcp_tool
                            logger.info("[MCP/SSE] Registered '%s' from %s", t.name, url)
            except Exception as exc:
                logger.warning("[MCP/SSE] Could not reach %s: %s", url, exc)

    async def _call_sse_tool(self, server_url: str, name: str, arguments: dict) -> str:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            return "(mcp package not installed)"
        try:
            async with sse_client(server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    parts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                        elif hasattr(block, "model_dump"):
                            parts.append(json.dumps(block.model_dump()))
                        else:
                            parts.append(str(block))
                    return "\n".join(parts) if parts else "(empty response)"
        except Exception as exc:
            logger.warning("[MCP/SSE] Tool call '%s' failed: %s", name, exc)
            return f"(MCP tool error: {exc})"


# ── Stdio transport (bundled arxiv server) ────────────────────────────────────

class StdioMCPClient(MCPClient):
    """
    Spawns src/mcp/arxiv_server.py as a subprocess and communicates
    over its stdin/stdout using JSON-RPC 2.0.
    No external server or network required.
    """

    def __init__(self) -> None:
        super().__init__()
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0

    def connect(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "src.mcp.arxiv_server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            # Initialize
            init_resp = self._rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "neuroresearch", "version": "1.0"},
            })
            if "error" in (init_resp or {}):
                raise RuntimeError(f"MCP init failed: {init_resp['error']}")

            # Notify initialized
            self._notify("notifications/initialized", {})

            # Discover tools
            tools_resp = self._rpc("tools/list", {})
            for t in (tools_resp or {}).get("result", {}).get("tools", []):
                mcp_tool = MCPTool(
                    name=t["name"],
                    description=t.get("description", t["name"]),
                    input_schema=t.get("inputSchema", {}),
                    client=self,
                )
                self._tools[t["name"]] = mcp_tool
                logger.info("[MCP/stdio] Registered '%s'", t["name"])

            logger.info("[MCP/stdio] Bundled arxiv server ready. Tools: %s", list(self._tools))

        except Exception as exc:
            logger.warning("[MCP/stdio] Bundled server failed to start (%s) — falling back.", exc)
            self._tools = {}
            if self._proc:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None

    def call_tool_sync(self, name: str, arguments: dict) -> str:
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if resp is None:
            return "(no response from MCP server)"
        if "error" in resp:
            return f"(MCP error: {resp['error'].get('message', resp['error'])})"
        content = resp.get("result", {}).get("content", [])
        parts   = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
        return "\n".join(parts) if parts else "(empty response)"

    def _rpc(self, method: str, params: dict) -> Optional[dict]:
        if not self._proc or self._proc.poll() is not None:
            return None
        self._req_id += 1
        request = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            return json.loads(line.strip()) if line.strip() else None
        except Exception as exc:
            logger.warning("[MCP/stdio] RPC error: %s", exc)
            return None

    def _notify(self, method: str, params: dict) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        notif = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(notif) + "\n")
            self._proc.stdin.flush()
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────

_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> Optional[MCPClient]:
    """Return the initialized MCPClient singleton, or None if MCP is disabled."""
    return _mcp_client


def init_mcp_client() -> Optional[MCPClient]:
    """
    Initialize the MCP client from settings. Priority:
    1. MCP_SERVER_URLS set → SSEMCPClient
    2. MCP_USE_BUNDLED_ARXIV=true → StdioMCPClient (bundled arxiv server)
    3. Neither → return None (ResearchAgent falls back to DuckDuckGo)
    """
    global _mcp_client

    try:
        from src.config.settings import get_settings
        s = get_settings()
        urls_raw           = s.MCP_SERVER_URLS
        use_bundled_arxiv  = s.MCP_USE_BUNDLED_ARXIV
    except Exception:
        urls_raw          = ""
        use_bundled_arxiv = False

    urls = [u.strip() for u in urls_raw.split() if u.strip()]

    if urls:
        logger.info("[MCP] Using SSE transport: %s", urls)
        client = SSEMCPClient(urls)
        client.connect()
        _mcp_client = client if client.has_tools() else None
        return _mcp_client

    if use_bundled_arxiv:
        logger.info("[MCP] Using bundled Arxiv MCP server (stdio)")
        client = StdioMCPClient()
        client.connect()
        _mcp_client = client if client.has_tools() else None
        return _mcp_client

    logger.info("[MCP] Not configured — ResearchAgent will use built-in DuckDuckGo search.")
    return None
