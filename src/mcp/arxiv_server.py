"""
Bundled Arxiv MCP Server — runs as a standalone process.

Exposes two tools over the MCP protocol (stdio transport):
  • search_arxiv(query, max_results=5)   — search arxiv.org papers
  • fetch_arxiv_abstract(arxiv_id)       — fetch a paper's abstract by ID

Usage (start as a subprocess, communicates over stdio):
    python -m src.mcp.arxiv_server

Or via the convenience launcher:
    python -m src.mcp.arxiv_server --port 8765   (starts HTTP/SSE bridge)

The ResearchAgent will automatically use this server when
MCP_SERVER_URLS=http://localhost:8765/sse is set in .env
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


# ── Arxiv API helpers ─────────────────────────────────────────────────────────

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


def _search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arxiv and return formatted results with titles, abstracts, URLs."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroResearch/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as exc:
        return f"Arxiv search failed: {exc}"

    try:
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", NS)
        if not entries:
            return f"No arxiv papers found for: {query}"

        results = []
        for entry in entries:
            title_el   = entry.find("atom:title", NS)
            summary_el = entry.find("atom:summary", NS)
            id_el      = entry.find("atom:id", NS)
            published_el = entry.find("atom:published", NS)
            authors    = [a.find("atom:name", NS).text
                          for a in entry.findall("atom:author", NS)
                          if a.find("atom:name", NS) is not None]

            title     = (title_el.text or "").strip().replace("\n", " ")
            abstract  = (summary_el.text or "").strip().replace("\n", " ")[:300]
            arxiv_url = (id_el.text or "").strip()
            published = (published_el.text or "")[:10]
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

            results.append(
                f"**{title}**\n"
                f"Authors: {author_str} ({published})\n"
                f"Abstract: {abstract}...\n"
                f"Source: {arxiv_url}"
            )

        return "\n\n".join(results)

    except ET.ParseError as exc:
        return f"Arxiv XML parse error: {exc}"


def _fetch_arxiv_abstract(arxiv_id: str) -> str:
    """Fetch a specific paper's full abstract by arxiv ID (e.g. '2303.08774')."""
    # Normalise: strip URL prefix if provided
    arxiv_id = arxiv_id.strip()
    if "/" in arxiv_id:
        arxiv_id = arxiv_id.split("/")[-1]

    params = urllib.parse.urlencode({
        "id_list": arxiv_id,
        "max_results": 1,
    })
    url = f"{ARXIV_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroResearch/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as exc:
        return f"Failed to fetch arxiv paper: {exc}"

    try:
        root  = ET.fromstring(xml_data)
        entry = root.find("atom:entry", NS)
        if entry is None:
            return f"No paper found for arxiv ID: {arxiv_id}"

        title   = (entry.find("atom:title",   NS).text or "").strip()
        summary = (entry.find("atom:summary", NS).text or "").strip()
        url_el  = entry.find("atom:id", NS)
        paper_url = (url_el.text or "").strip()

        return f"**{title}**\n\nAbstract:\n{summary}\n\nSource: {paper_url}"
    except ET.ParseError as exc:
        return f"XML parse error: {exc}"


# ── MCP Server (stdio transport) ──────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_arxiv",
        "description": (
            "Search arxiv.org for academic papers on a topic. "
            "Returns titles, authors, abstracts, and URLs for the most relevant papers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — topic, title keywords, or author name",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5, max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_arxiv_abstract",
        "description": (
            "Fetch the full abstract of a specific arxiv paper by its ID "
            "(e.g. '2303.08774' or full URL)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "Arxiv paper ID like '2303.08774' or full arxiv URL",
                },
            },
            "required": ["arxiv_id"],
        },
    },
]


def _handle_request(req: dict) -> dict:
    """Dispatch a single JSON-RPC 2.0 request and return the response dict."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": code, "message": message}}

    # MCP lifecycle
    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "arxiv-mcp-server", "version": "1.0.0"},
        })

    if method == "notifications/initialized":
        return None  # notification — no response

    if method == "tools/list":
        return ok({"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "search_arxiv":
            query       = arguments.get("query", "")
            max_results = int(arguments.get("max_results", 5))
            max_results = min(max_results, 10)
            text = _search_arxiv(query, max_results)
            return ok({"content": [{"type": "text", "text": text}]})

        if tool_name == "fetch_arxiv_abstract":
            arxiv_id = arguments.get("arxiv_id", "")
            text     = _fetch_arxiv_abstract(arxiv_id)
            return ok({"content": [{"type": "text", "text": text}]})

        return err(-32601, f"Unknown tool: {tool_name}")

    if method == "ping":
        return ok({})

    return err(-32601, f"Method not found: {method}")


def run_stdio_server() -> None:
    """
    Run the MCP server over stdio (stdin/stdout).
    Each line of stdin is a JSON-RPC request; each response is written to stdout.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req      = json.loads(line)
            response = _handle_request(req)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            err_resp = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


# ── HTTP/SSE bridge (makes stdio server accessible via HTTP) ──────────────────

def run_sse_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """
    Wrap the arxiv tools in a minimal HTTP server that speaks the SSE
    transport expected by MCPClient in client.py.

    This runs a FastAPI app that implements just enough of the MCP/SSE
    protocol for our client to discover and call the tools.
    """
    try:
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse, JSONResponse
        import asyncio
    except ImportError:
        print("FastAPI/uvicorn required for SSE bridge. pip install fastapi uvicorn")
        sys.exit(1)

    bridge = FastAPI(title="Arxiv MCP Server (SSE bridge)")

    # SSE endpoint — the client connects here first for the MCP handshake
    @bridge.get("/sse")
    async def sse_endpoint(request: Request):
        async def event_stream():
            # Send the endpoint URL so the client knows where to POST
            endpoint_url = f"http://{host}:{port}/messages"
            yield f"event: endpoint\ndata: {endpoint_url}\n\n"
            # Keep alive
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(15)
                yield ": keepalive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # Messages endpoint — the client POSTs JSON-RPC requests here
    @bridge.post("/messages")
    async def messages_endpoint(request: Request):
        body = await request.json()
        response = _handle_request(body)
        if response is None:
            return JSONResponse({})
        return JSONResponse(response)

    print(f"[Arxiv MCP Server] Listening on http://{host}:{port}/sse", flush=True)
    uvicorn.run(bridge, host=host, port=port, log_level="warning")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arxiv MCP Server")
    parser.add_argument("--port", type=int, default=0,
                        help="If set, run HTTP/SSE bridge on this port. "
                             "Otherwise run stdio server.")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.port:
        run_sse_server(host=args.host, port=args.port)
    else:
        run_stdio_server()
