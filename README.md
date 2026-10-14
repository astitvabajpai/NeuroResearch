---
title: NeuroResearch
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🧠 NeuroResearch — Self-Correcting Multi-Agent Research System

A production-grade AI research pipeline built with **LangGraph**, **FastAPI**, **Groq**, and **OpenRouter**.
Three specialised agents collaborate in a self-correcting loop to produce grounded, cited research
reports — powered by live Tavily web search, FAISS RAG retrieval, MCP tool calling, multi-rubric
critique, SSE streaming, session history, PDF export, and a built-in observability dashboard.

🔗 **GitHub:** [astitvabajpai/NeuroResearch](https://github.com/astitvabajpai/NeuroResearch)
🚀 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/johncenaqweewewwe/neuroResearch)

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         LangGraph Pipeline                            │
│                                                                        │
│  ┌──────────────────┐   ┌─────────────────┐   ┌────────────────────┐ │
│  │  Research Agent  │──►│  Writer Agent   │──►│  Critique Agent   │ │
│  │                  │   │                 │   │  (different model) │ │
│  │ • Tavily search  │   │ • Inline [cite] │   │ • 4-rubric scores │ │
│  │ • MCP page fetch │   │ • RAG-augmented │   │ • Source grounding│ │
│  │ • FAISS RAG index│   │ • 4 or 9 sects  │   │ • Actionable fix  │ │
│  └──────────────────┘   └─────────────────┘   └────────────────────┘ │
│           ▲                                              │              │
│           └──────────── revise if score < 0.85 ─────────┘              │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Tavily Search API  (live web search, full page content)
        │
        ▼
  MCP Fetch Tool     (fetches full article text via mcp-server-fetch)
        │
        ▼
  FAISS Vector Store (chunk → embed → index → retrieve top-k per section)
        │
        ▼
  Groq + OpenRouter  (7 verified free models, auto-rotates on rate limit)
        │
        ▼
  Observability
  • SQLite trace_events  (per-node latency, always on)
  • /observability page  (built-in timeline dashboard)
  • LangSmith / Langfuse (optional, via env vars)
```

---

## How It Works

| Agent | Role |
|---|---|
| 🔍 **Research Agent** | Runs Tavily search, fetches full pages via MCP, builds FAISS index, extracts findings with source URLs |
| ✍️ **Writer Agent** | Synthesises notes + RAG chunks into a structured report with `[Source](url)` inline citations |
| 🧐 **Critique Agent** | Scores 4 rubric dimensions, checks source grounding against retrieved URLs, returns actionable feedback |

The pipeline iterates until the critique score reaches the **quality threshold** (default `0.85`)
or `MAX_ITERATIONS` is reached. Each revision directly addresses the critique's specific feedback.

---

## Research Modes

| | Normal Mode | 🔬 Deep Research Mode |
|---|---|---|
| Tavily searches | 2 (topic + examples) | 3 (topic + advances + applications) |
| Pages fetched via MCP | up to 4 | up to 4 |
| RAG chunks retrieved | top-3 per section | top-5 per section |
| Report sections | 4 (Introduction → Conclusion) | 9 (Executive Summary → Conclusion) |
| Min words | 500 | 900+ |
| Findings extracted | 5 | 8–10 |
| Critique depth | Standard 4-rubric | Academic-level 4-rubric |

---

## Features

- 🌐 **Live Tavily search** — retrieves real current web results optimised for LLMs (DuckDuckGo fallback)
- 🔗 **MCP page fetch** — fetches full article text via `mcp-server-fetch` for richer source material
- 🧠 **FAISS RAG** — chunks and embeds fetched pages into a per-session vector store; Writer and Critique query it for relevant context
- 📎 **Inline citations** — every Key Finding ends with `[Source](url)` using only retrieved URLs
- 🧐 **4-rubric critique** — `factual_support`, `coverage`, `structure`, `clarity` scored separately (0–1)
- 🔀 **Cross-model critique** — Critique Agent uses `gpt-oss-120b` while Writer uses `gpt-oss-20b` to reduce self-preference scoring bias
- 🔁 **Source grounding check** — Critique Agent receives the retrieved URLs and verifies claim support
- 🤖 **7 models, 2 providers** — Groq (GPT OSS 20B/120B) + OpenRouter verified free models
- 🔄 **3-tier auto-rotation** — tries requested model → other Groq models → OpenRouter fallback
- 🛡️ **Per-user rate limiting** — `asyncio.Semaphore` limits each user to 1 concurrent run
- 🔒 **Input sanitization** — topics are length-capped (300 chars) and stripped of control characters; web content is injection-isolated
- 📊 **Built-in observability** — per-node latency, iteration count, rubric breakdown in `/observability`
- 🔬 **25-topic eval harness** — versioned JSON benchmark with 4 difficulty tiers and gold-fact coverage
- 🔐 **JWT + bcrypt auth** — per-user private sessions, 30-day tokens
- 📡 **Live SSE streaming** — real-time agent updates with latency and rubric scores
- 📄 **PDF export** — print-ready report with cover page and quality badge
- 🌙 **Dark UI** — vanilla HTML/CSS/JS, zero frontend framework
- ⚙️ **GitHub Actions CI** — ruff lint + pytest on every push

---

## Available Models

| Model ID | Name | Provider | Notes |
|---|---|---|---|
| `openai/gpt-oss-20b` | GPT OSS 20B ⭐ | Groq | Default Research + Writer |
| `openai/gpt-oss-120b` | GPT OSS 120B | Groq | Default Critique (different from Writer) |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | Nemotron 3 Ultra 550B | OpenRouter | Free, 1M ctx |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | Nemotron 3 Nano 30B | OpenRouter | Free, reasoning |
| `nvidia/nemotron-3.5-lightning:free` | Nemotron 3.5 Lightning | OpenRouter | Free, fast |
| `poolside/laguna-s-2.1:free` | Laguna S 2.1 | OpenRouter | Free |
| `openrouter/free` | Auto Free Router | OpenRouter | Auto-picks best available free model |

**Rotation order on failure:** requested model → other Groq models → OpenRouter free fallback

---

## Local Development

```bash
git clone https://github.com/astitvabajpai/NeuroResearch
cd NeuroResearch

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY and JWT_SECRET at minimum

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **http://localhost:8000** → register → start researching
- **http://localhost:8000/observability** → trace timeline dashboard
- **http://localhost:8000/mcp/tools** → active MCP tools
- **http://localhost:8000/health** → status (search provider, RAG availability)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free key from [console.groq.com](https://console.groq.com/keys) |
| `JWT_SECRET` | ✅ | Any random string — signs auth tokens |
| `TAVILY_API_KEY` | recommended | Free key from [app.tavily.com](https://app.tavily.com) — no card, richer search results |
| `OPENROUTER_API_KEY` | recommended | Free key from [openrouter.ai](https://openrouter.ai) — no card, 5 extra fallback models |
| `HF_MODEL_ID` | optional | Default model (default: `openai/gpt-oss-20b`) |
| `MAX_ITERATIONS` | optional | Max pipeline loops (default: `3`) |
| `QUALITY_THRESHOLD` | optional | Score to stop iterating (default: `0.85`) |
| `CORS_ORIGINS` | optional | Comma-separated allowed origins (default: `*`) |
| `LANGCHAIN_TRACING_V2` | optional | `true` to enable LangSmith |
| `LANGCHAIN_API_KEY` | optional | LangSmith API key |
| `LANGFUSE_PUBLIC_KEY` | optional | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | optional | Langfuse secret key |

---

## RAG Pipeline

On every research iteration:
1. Tavily returns search results with source URLs
2. The Research Agent fetches full page content for each URL (MCP → HTTP fallback)
3. Pages are chunked (~300 chars with overlap) and embedded using `sentence-transformers/all-MiniLM-L6-v2`
4. Chunks are stored in a per-session FAISS index (isolated per run, cleared on completion)
5. The Writer Agent queries the index for the most relevant chunks per section before writing
6. The Critique Agent receives the retrieved URLs to verify that citations in the report are grounded

> **Note:** RAG is automatically disabled on HuggingFace Spaces due to memory constraints. All other features work fully.

---

## MCP Tool Calling

The project uses the [Model Context Protocol](https://modelcontextprotocol.io) to call external tools:

| Tool | Server | What it does |
|---|---|---|
| `mcp_fetch` | `mcp-server-fetch` (via uvx) | Fetches full text of any URL — richer than DuckDuckGo snippets |
| `mcp_brave_search` | `mcp-server-brave-search` | Brave Search API (requires `BRAVE_API_KEY`) |
| `mcp_arxiv_search` | `mcp-server-arxiv` | Academic paper search (opt-in) |

MCP tools are registered at startup and exposed at `GET /mcp/tools`. They fall back gracefully if `uvx` or the `mcp` package is unavailable.

> **Note:** MCP subprocess calling is disabled on HuggingFace Spaces. The HTTP fetcher is used instead.

---

## Observability

### Built-in dashboard (always on)

Every LangGraph node execution is stored in SQLite (`trace_events`).
View at **http://localhost:8000/observability**:

- Per-session: total latency, critique score + rubric breakdown, iteration count, error count
- Per-node: which agent ran, how long it took, what it read/wrote
- Per-agent latency bars: avg/max for research, write, critique

### LangSmith (optional)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=neuroresearch
```

### Langfuse (optional, self-hostable)

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Eval Harness

```bash
# Run all 25 benchmark topics
python -m src.evals.harness

# Quick smoke test (easy topics only)
python -m src.evals.harness --difficulty easy --no-save

# Score sessions already in the DB
python -m src.evals.harness --from-db

# Single topic
python -m src.evals.harness --topic "transformer architecture"

# Include LLM faithfulness judge
python -m src.evals.harness --faithfulness
```

Reports saved to `data/eval_results/YYYY-MM-DD_<model>_<mode>.json`

| Tier | Topics | Tests |
|---|---|---|
| `easy` (5) | Transformers, CRISPR, RAG, Fine-tuning, Gradient descent | Core facts |
| `medium` (11) | Climate, AlphaFold, RLHF, Diffusion models, LangGraph… | Multi-source synthesis |
| `hard` (5) | Quantum crypto, MoE, Mechanistic interpretability… | Deep mode, 9-section |
| `adversarial` (4) | Nonsense topic, consciousness, bias, AGI economics | Graceful degradation |

---

## Where It Failed — Real Failure Analysis

### 1. Note duplication on iteration 2+
**Root cause:** `ResearchAgent` manually prepended existing notes AND LangGraph's `operator.add` reducer also accumulated them — double-counting on every loop.
**Fix:** Return only `[notes_str]`; the reducer handles accumulation. Cap to last 2 note batches in the Writer to prevent context overflow.

### 2. passlib + bcrypt incompatibility on Python 3.13
**Root cause:** `passlib 1.7.4` uses an internal bcrypt API that changed in `bcrypt 4.x`.
**Fix:** Removed passlib entirely, use `bcrypt.hashpw/checkpw` directly.

### 3. FastAPI/Starlette version conflict
**Root cause:** Installing `mcp` upgraded Starlette to 1.3.1 which broke FastAPI 0.115.5.
**Fix:** Upgraded FastAPI to 0.116+, replaced `@app.on_event("startup")` with `@asynccontextmanager lifespan`.

### 4. Groq rate limiting during multi-iteration runs
**Root cause:** 3-iteration run fires 9 LLM calls, hitting Groq's per-minute per-model limit.
**Fix:** 3-tier rotation — Groq models → OpenRouter free fallback. Effectively multiplies free quota.

### 5. Qwen 3.6 27B removed from Groq free tier
**Root cause:** `qwen/qwen3.6-27b` returned HTTP 404 — Groq removed it from the free tier.
**Fix:** Replaced with `gpt-oss-120b` as the cross-model critique default + 5 OpenRouter free models as fallbacks.

### 6. sentence-transformers / huggingface_hub version conflict
**Root cause:** `sentence-transformers 3.4.x` requires `huggingface_hub<2.0` but the project uses `2.2.0`.
**Fix:** Use `langchain_huggingface.HuggingFaceEmbeddings` directly (bypasses the conflicting import path in sentence-transformers).

### 7. MCP sync call fails in thread pool
**Root cause:** LangGraph runs agents in thread pool executors which have no asyncio event loop. `asyncio.get_event_loop()` raises there.
**Fix:** Always use `asyncio.new_event_loop()` in the sync MCP wrapper — creates a fresh loop per call, safe in any thread context.

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py          Abstract BaseAgent
│   │   ├── research_agent.py      Tavily + MCP fetch + FAISS indexing + LLM extraction
│   │   ├── writer_agent.py        RAG-augmented report generation (4 or 9 sections)
│   │   └── critique_agent.py      4-rubric scoring with source grounding check
│   ├── graph/
│   │   └── research_graph.py      LangGraph StateGraph + conditional edges
│   ├── state/
│   │   └── research_state.py      ResearchState TypedDict (session_id, rag_enabled, mcp_tools…)
│   ├── tools/
│   │   ├── llm.py                 Groq + OpenRouter — 7 models, 3-tier auto-rotation
│   │   ├── search_tool.py         Tavily primary, DuckDuckGo fallback
│   │   ├── fetcher.py             HTTP page fetcher (BeautifulSoup, used as MCP fallback)
│   │   └── vector_store.py        FAISS per-session RAG index (langchain_huggingface embeddings)
│   ├── mcp/
│   │   ├── client.py              Async MCP client (stdio transport)
│   │   ├── servers.py             MCP server registry (fetch, brave-search, arxiv)
│   │   └── tool_registry.py       Wraps MCP tools as LangChain Tools
│   ├── evals/
│   │   ├── dataset.py             25-topic benchmark with difficulty + gold facts
│   │   ├── metrics.py             Deterministic + LLM-judge metrics
│   │   └── harness.py             CLI runner, versioned JSON reports
│   ├── observability/
│   │   └── tracing.py             LangSmith + Langfuse setup
│   ├── api.py                     FastAPI + SSE + rate limiting + /mcp/tools + /observability
│   ├── auth.py                    JWT + bcrypt
│   └── database.py                SQLite: users, sessions, trace_events
├── frontend/
│   ├── index.html                 Main research UI (7-model selector, rubric display)
│   ├── login.html                 Auth page
│   └── observability.html         Trace timeline dashboard
├── tests/
│   ├── test_graph_routing.py      State schema, routing logic, model defaults (8 tests)
│   └── test_auth.py               Auth endpoints, topic validation (9 tests)
├── .github/
│   └── workflows/
│       └── ci.yml                 Ruff lint + pytest on every push
├── .env.example
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · SQLite · python-jose · bcrypt

**LLM:** Groq (GPT OSS 20B/120B) · OpenRouter free tier (Nemotron, Laguna, Auto-router)

**Search:** Tavily API · DuckDuckGo (fallback)

**RAG:** FAISS · sentence-transformers (`all-MiniLM-L6-v2`) · langchain-huggingface

**MCP:** mcp SDK · uvx · mcp-server-fetch · mcp-server-brave-search (optional)

**Observability:** SQLite trace_events · LangSmith · Langfuse · built-in `/observability`

**Eval:** 25-topic benchmark harness · deterministic + LLM-judge metrics

**Frontend:** Vanilla HTML · CSS · JavaScript (no framework)

**Deployment:** Docker · GitHub Actions CI (ruff + pytest)
