"""
MCP Client — connects to MCP servers via stdio transport.

Wraps the mcp SDK's StdioServerParameters so each server is started as a
subprocess and communicated with over stdin/stdout using JSON-RPC.

Graceful degradation:
- If `mcp` package is not installed → returns None
- If running on HuggingFace Spaces → returns None (subprocess spawning blocked)
- If uvx is not found → returns None
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# HF Spaces blocks subprocess spawning — disable MCP there
_ON_HF_SPACES = os.environ.get("SPACE_ID") is not None


async def call_mcp_tool(
    server_command: list[str],
    tool_name: str,
    arguments: dict,
    env: Optional[dict] = None,
) -> Optional[str]:
    """
    Start an MCP server as a subprocess, call one tool, return its text output.
    Returns None on HF Spaces, missing mcp package, or any failure.
    """
    if _ON_HF_SPACES:
        logger.debug("[MCP] Disabled on HuggingFace Spaces")
        return None

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning("[MCP] mcp package not installed — MCP tools disabled")
        return None

    params = StdioServerParameters(
        command=server_command[0],
        args=server_command[1:],
        env=env,
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                # Extract text content from the result
                if result and result.content:
                    texts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                    return "\n".join(texts) if texts else None
                return None
    except Exception as exc:
        logger.warning("[MCP] Tool call failed (%s / %s): %s", tool_name, server_command[0], exc)
        return None


def call_mcp_tool_sync(
    server_command: list[str],
    tool_name: str,
    arguments: dict,
    env: Optional[dict] = None,
) -> Optional[str]:
    """
    Synchronous wrapper around call_mcp_tool for use in non-async agent code.
    Always creates a fresh event loop — safe to call from any thread.
    """
    try:
        # Always create a new event loop — agents run in thread pool workers
        # that never have a running loop, so asyncio.get_event_loop() fails.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                call_mcp_tool(server_command, tool_name, arguments, env)
            )
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[MCP] Sync call failed: %s", exc)
        return None
