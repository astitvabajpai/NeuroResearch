# 🧠 NeuroResearch — Self-Correcting Multi-Agent Research System

A production-grade AI research pipeline built with **LangGraph**, **FastAPI**, and **Groq AI**.
Three specialised agents collaborate in a self-correcting loop to produce high-quality research
reports on any topic — with live streaming, session history, PDF export, and a built-in
observability dashboard.

🔗 **GitHub:** [astitvabajpai/NeuroResearch](https://github.com/astitvabajpai/NeuroResearch)

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                   LangGraph Pipeline                  │
│                                                       │
│  Research Agent ──► Writer Agent ──► Critique Agent   │
│       │                                     │         │
│       │◄──── revise if score < threshold ───┘         │
└──────────────────────────────────────────────────────-┘
       │
       ▼
  DuckDuckGo Web Search
  (live internet search on every iteration)
       │
       ▼
  Groq API  (5 free models, auto-rotates on rate limit)
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
| 🔍 **Research Agent** | Searches the web (DuckDuckGo), extracts key findings using Groq LLM |
| ✍️ **Writer Agent** | Synthesises findings into a structured markdown report, uses critique feedback |
| 🧐 **Critique Agent** | Scores the draft 0–1 on accuracy, completeness, and clarity |

The pipeline iterates until the critique score reaches the **quality threshold** (default 0.95)
or `MAX_ITERATIONS` is reached. The Writer Agent receives the Critique Agent's feedback on each
iteration and directly addresses it in the next draft.

---

## Research Modes

| | Normal Mode | 🔬 Deep Research Mode |
|---|---|---|
| Web searches | 2 (topic + overview) | 3 (topic + latest advances + applications) |
| Report sections | 4 (Introduction → Conclusion) | 9 (Executive Summary → Conclusion) |
| Min words | 500 | 900+ |
| Max tokens | 1 024 | 2 048 |
| Findings extracted | 5 | 8–10 |
| Critique depth | Standard | Academic-level |

---

## Features

- 🤖 **5 free Groq models** — Llama 3.3 70B, Qwen 3.6 27B, GPT OSS 120B/20B, Llama 3.1 8B
- 🔄 **Auto-rotation** — if one model is rate-limited, automatically tries the next
- 🌐 **Live web search** — DuckDuckGo searches the real internet on every research iteration
- 📊 **Built-in observability** — per-node latency timeline at `/observability`, no API key needed
- 🔬 **25-topic eval harness** — versioned JSON benchmark with difficulty tiers
- 🔐 **JWT auth** — per-user private sessions
- 🤖 **Per-agent model selection** — choose different Groq models per agent
- 📡 **Live SSE streaming** — watch each agent work in real time with latency display
- 📄 **PDF export** — print-ready report with cover page and quality badge
- 🌙 **Dark UI** — vanilla HTML/CSS/JS, zero frontend framework

---

## Available Models

All models are **free** on Groq and support deep research (2 048 tokens):

| Model ID | Name | Best for |
|---|---|---|
| `llama-3.3-70b-versatile` | Llama 3.3 70B ⭐ | Default — best all-round quality |
| `qwen/qwen3.6-27b` | Qwen 3.6 27B | Most detailed, longest outputs |
| `openai/gpt-oss-120b` | GPT OSS 120B | Most powerful reasoning |
| `openai/gpt-oss-20b` | GPT OSS 20B | Balanced speed/quality |
| `llama-3.1-8b-instant` | Llama 3.1 8B | Fastest responses |

If a model is rate-limited, the pipeline automatically tries the next one in this order.

---

## Local Development

```bash
git clone https://github.com/astitvabajpai/NeuroResearch
cd NeuroResearch

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY and JWT_SECRET (both required)

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** → register → start researching.
Open **http://localhost:8000/observability** → view trace timeline.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free key from [console.groq.com](https://console.groq.com/keys) |
| `JWT_SECRET` | ✅ | Any random string — signs auth tokens |
| `HF_MODEL_ID` | optional | Default model ID (default: `llama-3.3-70b-versatile`) |
| `MAX_ITERATIONS` | optional | Max pipeline loops (default: 3) |
| `QUALITY_THRESHOLD` | optional | Score to stop at (default: 0.95) |
| `LANGCHAIN_TRACING_V2` | optional | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | optional | LangSmith API key |
| `LANGFUSE_PUBLIC_KEY` | optional | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | optional | Langfuse secret key |

---

## Observability

### Built-in dashboard (always on, no API key needed)

Every LangGraph node execution is timed and stored in SQLite (`trace_events` table).
View the timeline at **http://localhost:8000/observability**:

- Per-session stats: total latency, critique score, iterations, error count
- Per-node timeline: which agent ran when, how long it took, what it read/wrote
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

# Quick smoke test (5 easy topics only)
python -m src.evals.harness --difficulty easy --no-save

# Score sessions already in the DB
python -m src.evals.harness --from-db

# Single ad-hoc topic
python -m src.evals.harness --topic "transformer architecture"
```

Reports saved to `data/eval_results/YYYY-MM-DD_<model>_<mode>.json`

### Benchmark difficulty tiers

| Tier | Topics | Tests |
|---|---|---|
| `easy` (5) | Transformers, CRISPR, RAG, Fine-tuning, Gradient descent | Core facts, known terminology |
| `medium` (11) | Climate, AlphaFold, RLHF, Diffusion models, LangGraph, … | Multi-source synthesis |
| `hard` (5) | Quantum crypto, MoE, Mechanistic interpretability, … | Deep mode, 9-section structure |
| `adversarial` (4) | Nonsense topic, consciousness, bias, AGI economics | Graceful degradation |

---

## Where It Failed — Real Failure Analysis

### 1. Note duplication on iteration 2+
**Root cause:** `ResearchAgent` manually prepended existing notes AND LangGraph's `operator.add`
reducer also accumulated them — double-counting on every loop.
**Fix:** Return only the new note `[notes_str]`; the reducer handles accumulation automatically.

### 2. passlib + bcrypt incompatibility on Python 3.13
**Root cause:** `passlib 1.7.4` uses an internal bcrypt API that changed in `bcrypt 4.x`,
causing a crash on every login/register call.
**Fix:** Removed passlib entirely, use `bcrypt.hashpw/checkpw` directly.

### 3. FastAPI/Starlette version conflict
**Root cause:** Installing `mcp` upgraded Starlette to 1.3.1 which broke FastAPI 0.115.5.
**Fix:** Upgraded FastAPI to 0.140+ and replaced deprecated `@app.on_event("startup")`
with the modern `@asynccontextmanager lifespan` pattern.

### 4. Groq rate limiting during multi-iteration runs
**Root cause:** A 3-iteration run fires 9 LLM calls (3 agents × 3 iterations), hitting
Groq's per-minute per-model limit.
**Fix:** Auto-rotation across 5 Groq models — if one is rate-limited the next is tried
immediately. Effectively 5× the free quota.

### 5. Low critique scores from miscalibrated prompt
**Root cause:** The Critique Agent had no scoring rubric, so it scored inconsistently —
often giving 0.5–0.6 to solid reports.
**Fix:** Added explicit calibration notes to the critique prompt defining what score range
corresponds to what quality level.

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py        Abstract BaseAgent
│   │   ├── research_agent.py    Web search + LLM extraction
│   │   ├── writer_agent.py      Report generation (4 or 9 sections)
│   │   └── critique_agent.py    Calibrated quality scoring (0–1)
│   ├── graph/
│   │   └── research_graph.py    LangGraph StateGraph + conditional edges
│   ├── state/
│   │   └── research_state.py    ResearchState TypedDict
│   ├── tools/
│   │   ├── llm.py               Groq API — 5 models, auto-rotation
│   │   └── search_tool.py       DuckDuckGo web search (ddgs)
│   ├── evals/
│   │   ├── dataset.py           25-topic benchmark with difficulty + gold facts
│   │   ├── metrics.py           Coverage, faithfulness, retrieval metrics
│   │   └── harness.py           CLI runner, versioned JSON reports
│   ├── observability/
│   │   └── tracing.py           LangSmith + Langfuse setup
│   ├── api.py                   FastAPI + SSE streaming + /observability route
│   ├── auth.py                  JWT + bcrypt
│   └── database.py              SQLite: users, sessions, trace_events
├── frontend/
│   ├── index.html               Main research UI
│   ├── login.html               Auth page
│   └── observability.html       Trace timeline dashboard
├── .env.example
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · SQLite · python-jose · bcrypt

**LLM:** Groq API (free tier) — Llama 3.3, Qwen 3.6, GPT OSS, Llama 3.1

**Search:** DuckDuckGo (ddgs) with retry/backoff and instant-answer fallback

**Observability:** SQLite trace_events · LangSmith · Langfuse · built-in `/observability` page

**Eval:** 25-topic benchmark harness with deterministic + LLM-judge metrics

**Frontend:** Vanilla HTML · CSS · JavaScript (no framework)

**Deployment:** Docker
