# 🧠 NeuroResearch — Self-Correcting Multi-Agent Research System

A production-grade AI research pipeline built with **LangGraph**, **FastAPI**, and **HuggingFace Inference API**. Three specialised agents collaborate in a self-correcting loop, now powered by a real **MCP tool server** for structured academic retrieval.

🔗 **Live Demo:** [HuggingFace Spaces](https://johncenaqweewewwe-neuroresearch.hf.space)  
🔗 **GitHub:** [astitvabajpai/NeuroResearch](https://github.com/astitvabajpai/NeuroResearch)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                  │
│                                                      │
│  Research Agent ──► Writer Agent ──► Critique Agent  │
│       │                                    │         │
│       │◄──────── revise if score < 0.8 ────┘         │
│       │                                              │
│  ┌────┴──────┐                                       │
│  │ MCP Client│  (auto-selects tool at startup)       │
│  └────┬──────┘                                       │
└───────┼─────────────────────────────────────────────-┘
        │  MCP Protocol (stdio / SSE)
        ▼
┌───────────────────────┐   ┌──────────────────────┐
│  Bundled Arxiv MCP    │   │  External MCP Server  │
│  (zero-config, stdio) │   │  (set MCP_SERVER_URLS) │
│  • search_arxiv       │   │  GitHub / filesystem  │
│  • fetch_abstract     │   │  or your own server   │
└───────────────────────┘   └──────────────────────┘
        │
        ▼
Observability layer
  • SQLite trace_events  (per-node latency, errors)
  • LangSmith            (set LANGCHAIN_TRACING_V2=true)
  • Langfuse             (set LANGFUSE_* keys)
  • /observability page  (built-in timeline dashboard)
        │
        ▼
Eval harness  (python -m src.evals.harness)
  • 25-topic versioned benchmark
  • Retrieval precision / recall vs gold facts
  • LLM faithfulness judge
  • data/eval_results/YYYY-MM-DD_model_mode.json
```

---

## How It Works

| Agent | Role |
|---|---|
| 🔍 **Research Agent** | Calls MCP tool → structured arxiv results; falls back to DuckDuckGo |
| ✍️ **Writer Agent** | Synthesises notes into a structured markdown report |
| 🧐 **Critique Agent** | Scores draft 0–1 on accuracy, completeness, clarity |

The pipeline iterates until the critique score reaches **0.8** or `MAX_ITERATIONS` is hit.

---

## Research Modes

| | Normal | 🔬 Deep Research |
|---|---|---|
| MCP searches | 1 | 2 (topic + "latest advances") |
| Report sections | 4 | 9 (Executive Summary → Conclusion) |
| Min words | 400 | 800+ |
| Max tokens | 1 024 | 2 048 |
| Critique depth | Basic | Academic-level |

---

## Features

- 🔌 **MCP tool integration** — bundled Arxiv server (zero-config) or plug in any MCP-compatible server
- 📊 **Built-in observability** — per-node latency timeline at `/observability`, no external service needed
- 🔬 **25-topic eval harness** — versioned JSON reports, retrieval precision/recall, LLM faithfulness judge
- 🔐 **JWT auth** — per-user private sessions
- 🤖 **Per-agent model selection** — mix Llama 3, Qwen 2.5, Mistral, Phi-3 per agent
- 📡 **Live SSE streaming** — watch each agent run in real time with latency display
- 📄 **PDF export** — print-ready report with cover page
- 🌙 **Dark UI** — vanilla HTML/CSS/JS, zero frontend framework

---

## Eval Results

Run with: `python -m src.evals.harness --faithfulness`

Reports saved to `data/eval_results/YYYY-MM-DD_<model>_<mode>.json`

| Metric | Description |
|---|---|
| `overall` | Weighted composite (critique 25%, coverage 15%, retrieval 20%, faithfulness 20%, …) |
| `retrieval_precision` | Fraction of gold facts found in research notes |
| `retrieval_recall` | Fraction of expected source domains present in notes |
| `faithfulness` | LLM-judge: are report claims grounded in retrieved content? |
| `section_coverage` | Required ## headings present |
| `iteration_efficiency` | Convergence speed (1.0 = first iteration, 0.0 = never converged) |

### Difficulty tiers in the benchmark

| Tier | Topics | What it tests |
|---|---|---|
| `easy` (5) | Transformers, CRISPR, Gradient descent, Fine-tuning, RAG | Core facts and well-cited papers |
| `medium` (9) | Climate, AlphaFold, Federated learning, RLHF, Diffusion, … | Multi-source synthesis |
| `hard` (4) | Quantum crypto, MoE, Mechanistic interp., AI safety | Deep mode, 9-section structure, 800+ words |
| `adversarial` (4) | Nonsense topic, consciousness, AGI economics, bias | Graceful degradation, no hallucination |

Run only easy topics to smoke-test a new model quickly:
```bash
python -m src.evals.harness --difficulty easy --model Qwen/Qwen2.5-7B-Instruct
```

---

## MCP Integration

### Default (bundled Arxiv server — no setup)

Set in `.env`:
```
MCP_USE_BUNDLED_ARXIV=true
```

The server spawns as a subprocess. Tools available:
- `search_arxiv(query, max_results=5)` — returns titles, authors, abstracts, arxiv URLs
- `fetch_arxiv_abstract(arxiv_id)` — fetches a specific paper by ID

### External MCP server

```
MCP_SERVER_URLS=http://localhost:3000/sse
```

Start the GitHub MCP server:
```bash
npx @modelcontextprotocol/server-github
```

Or the filesystem server:
```bash
npx @modelcontextprotocol/server-filesystem .
```

The Research Agent auto-discovers all tools and prefers any tool named `web_search`, then any tool containing "search", then the first available tool.

---

## Observability

### Built-in (always on)

Every LangGraph node execution is timed and stored in `trace_events` (SQLite). View it at **http://localhost:8000/observability** — no API keys needed.

The timeline shows:
- Per-node latency (research / write / critique)
- Which state keys were read/written
- Errors per node
- Convergence speed across iterations

### LangSmith (optional)

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=neuroresearch
```

Auto-instruments all LangChain/LangGraph calls. Every run gets a named trace (`research:<topic>`) with tags and the final critique score pushed back as a score event.

### Langfuse (optional, self-hostable)

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Where It Failed — Real Failure Analysis

These failure patterns were discovered while building the eval harness:

### 1. Note duplication on iteration 2+
**Symptom:** Research notes doubled on each loop iteration — the Writer received 2× the context on iteration 2, 4× on iteration 3, causing prompt truncation and incoherent drafts.  
**Root cause:** `ResearchAgent` manually prepended existing notes *and* LangGraph's `operator.add` reducer also accumulated them — double-counting.  
**Fix:** Return only the new note `[notes_str]`; let the reducer handle accumulation.

### 2. Critique score never converging on adversarial topics
**Symptom:** Topics like "xyzplexor neural optimization" hit `max_iterations` with score ~0.3. The fallback search returned a "use training knowledge" message, which the LLM used to generate plausible-sounding but completely fabricated content. The Critique Agent scored this low, triggering endless revisions.  
**What we learned:** Rate-limited or failed searches are a silent failure mode. The fix is monitoring `source_density` in the eval harness — if it's 0.0, the search tool returned no URLs, meaning no real content was retrieved.

### 3. MCP module-level import crash
**Symptom:** App failed to start when `HF_API_TOKEN` was not set because `hg_llm.py` called `get_settings()` at module level, which validated the token on import.  
**Fix:** Made all settings reads lazy — `get_settings()` is called inside functions, never at module top-level.

### 4. FastAPI/Starlette version conflict
**Symptom:** `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` after installing the `mcp` package, which pulled in Starlette 1.3.1 while FastAPI 0.115.5 expected 0.41.3.  
**Fix:** Replaced deprecated `@app.on_event("startup")` with the modern `@asynccontextmanager lifespan` pattern, then upgraded FastAPI to 0.140+ which is compatible with Starlette 1.x.

---

## Local Development

```bash
git clone https://github.com/astitvabajpai/NeuroResearch
cd NeuroResearch

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set HF_API_TOKEN (required) and optionally MCP_USE_BUNDLED_ARXIV=true

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** → login → start researching.  
Open **http://localhost:8000/observability** → view trace timeline.

### Run the eval harness

```bash
# Quick smoke test (5 easy topics, no faithfulness)
python -m src.evals.harness --difficulty easy --no-save

# Full benchmark with faithfulness scoring
python -m src.evals.harness --faithfulness

# Single topic
python -m src.evals.harness --topic "transformer architecture" --faithfulness

# Score sessions already saved in the DB
python -m src.evals.harness --from-db

# Compare models
python -m src.evals.harness --difficulty easy --model mistralai/Mistral-7B-Instruct-v0.3
python -m src.evals.harness --difficulty easy --model Qwen/Qwen2.5-7B-Instruct
```

Results are saved to `data/eval_results/YYYY-MM-DD_<model>_<mode>.json`.

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # Abstract BaseAgent
│   │   ├── research_agent.py   # MCP tool selection + search
│   │   ├── writer_agent.py     # Report generation (4 or 9 sections)
│   │   └── critique_agent.py   # Quality scoring (0–1)
│   ├── graph/
│   │   └── research_graph.py   # LangGraph StateGraph + conditional edges
│   ├── state/
│   │   └── research_state.py   # ResearchState TypedDict
│   ├── mcp/
│   │   ├── client.py           # SSE + stdio MCP client, auto-selects transport
│   │   └── arxiv_server.py     # Bundled Arxiv MCP server (stdio, zero-config)
│   ├── tools/
│   │   ├── hg_llm.py           # HF Inference API → LangChain BaseChatModel
│   │   └── search_tool.py      # DuckDuckGo fallback with retry/backoff
│   ├── evals/
│   │   ├── dataset.py          # 25-topic benchmark with difficulty + gold facts
│   │   ├── metrics.py          # Retrieval precision/recall, faithfulness judge, coverage
│   │   └── harness.py          # CLI runner, versioned JSON reports
│   ├── observability/
│   │   └── tracing.py          # LangSmith + Langfuse setup
│   ├── api.py                  # FastAPI app, SSE streaming, /observability route
│   ├── auth.py                 # JWT + bcrypt
│   └── database.py             # SQLite: users, sessions, trace_events
├── frontend/
│   ├── index.html              # Main research UI
│   ├── login.html              # Auth page
│   └── observability.html      # Trace timeline dashboard
├── data/
│   └── eval_results/           # Versioned eval JSON reports
├── .env.example
├── Dockerfile
└── requirements.txt
```

---

## Available Models

| Model | Tag | Notes |
|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⭐ Recommended | Requires HF access request |
| `Qwen/Qwen2.5-7B-Instruct` | Strong | Best eval scores in testing |
| `mistralai/Mistral-7B-Instruct-v0.3` | Fast | Lowest latency |
| `microsoft/Phi-3-mini-4k-instruct` | Lightweight | Good for adversarial topics |
| `HuggingFaceH4/zephyr-7b-beta` | — | |
| `google/gemma-2-2b-it` | Small | |

---

## Deploying to HuggingFace Spaces

1. Create a Space → **Docker SDK**
2. Push this repo to the Space's git remote
3. Add secrets in **Space → Settings → Variables and secrets**:

| Variable | Required | Description |
|---|---|---|
| `HF_API_TOKEN` | ✅ | HuggingFace token (read access) |
| `JWT_SECRET` | ✅ | Any random secret string |
| `MCP_USE_BUNDLED_ARXIV` | recommended | `true` — enables arxiv search |
| `HF_MODEL_ID` | optional | Default model |

4. Enable **Persistent Storage** so SQLite survives restarts.

---

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · SQLite · JWT (python-jose) · passlib  
**LLM:** HuggingFace Inference API via `router.huggingface.co`  
**Search:** Bundled Arxiv MCP server (stdio) · DuckDuckGo fallback  
**MCP:** `mcp` Python SDK · SSE + stdio transports  
**Observability:** SQLite trace_events · LangSmith · Langfuse  
**Eval:** Custom harness with LLM-judge faithfulness scoring  
**Frontend:** Vanilla HTML · CSS · JavaScript (no framework)  
**Deployment:** Docker on HuggingFace Spaces
