---
title: NeuroResearch
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🧠 NeuroResearch — Self-Correcting Multi-Agent Research System

A production-grade AI research pipeline built with **LangGraph**, **FastAPI**, and **Groq AI**.
Three specialised agents collaborate in a self-correcting loop to produce grounded, cited research
reports on any topic — with live web retrieval, inline source citations, multi-rubric critique,
SSE streaming, session history, PDF export, and a built-in observability dashboard.

🔗 **GitHub:** [astitvabajpai/NeuroResearch](https://github.com/astitvabajpai/NeuroResearch)
🚀 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/johncenaqweewewwe/neuroResearch)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Pipeline                        │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │  Research   │───►│   Writer    │───►│       Critique       │ │
│  │   Agent     │    │   Agent     │    │       Agent          │ │
│  │             │    │             │    │  (different model)   │ │
│  │ • DDG search│    │ • Inline    │    │ • 4-rubric scoring   │ │
│  │ • URL notes │    │   citations │    │ • Source grounding   │ │
│  │ • Gold facts│    │ • Structured│    │ • Actionable feedback│ │
│  └─────────────┘    └─────────────┘    └──────────────────────┘ │
│         ▲                                          │              │
│         └──────── revise if score < threshold ─────┘              │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
  DuckDuckGo Web Search  (live retrieval on every iteration)
       │
       ▼
  Groq API  (3 free models, auto-rotates on rate limit)
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
| 🔍 **Research Agent** | Searches the web (DuckDuckGo), extracts key findings with source URLs |
| ✍️ **Writer Agent** | Synthesises findings into a structured report with inline citations |
| 🧐 **Critique Agent** | Scores 4 rubric dimensions, checks source grounding, returns actionable feedback |

The pipeline iterates until the critique score reaches the **quality threshold** (default 0.75)
or `MAX_ITERATIONS` is reached. The Writer Agent receives the Critique Agent's specific feedback
on each iteration and directly addresses it in the next draft.

---

## Research Modes

| | Normal Mode | 🔬 Deep Research Mode |
|---|---|---|
| Web searches | 2 (topic + overview) | 3 (topic + latest advances + applications) |
| Report sections | 4 (Introduction → Conclusion) | 9 (Executive Summary → Conclusion) |
| Min words | 500 | 900+ |
| Max tokens | 1 024 | 2 048 |
| Findings extracted | 5 | 8–10 |
| Critique depth | Standard 4-rubric | Academic-level 4-rubric |

---

## Features

- 🌐 **Live web retrieval** — DuckDuckGo searches the real internet on every research iteration
- 📎 **Inline citations** — every report section links claims back to retrieved source URLs
- 🧐 **Multi-rubric critique** — 4 separate scores (factual support, coverage, structure, clarity) instead of one number
- 🔀 **Cross-model critique** — Critique Agent uses a different model than the Writer to reduce self-preference bias
- 🔁 **Source grounding check** — Critique Agent verifies claims are supported by retrieved text, not just well-written
- 🤖 **3 free Groq models** — Qwen 3.6 27B, GPT OSS 120B, GPT OSS 20B with auto-rotation
- 🔄 **Auto-rotation** — if one model is rate-limited, automatically tries the next
- 📊 **Built-in observability** — per-node latency timeline at `/observability`, no API key needed
- 🔬 **25-topic eval harness** — versioned JSON benchmark with difficulty tiers and gold-fact coverage
- 🔐 **JWT auth** — per-user private sessions with bcrypt password hashing
- 🤖 **Per-agent model selection** — choose different Groq models per agent in the UI
- 📡 **Live SSE streaming** — watch each agent work in real time with latency display
- 🛡️ **Per-user rate limiting** — in-memory semaphore prevents concurrent run abuse
- 📄 **PDF export** — print-ready report with cover page and quality badge
- 🌙 **Dark UI** — vanilla HTML/CSS/JS, zero frontend framework

---

## Available Models

All models are **free** on Groq and support deep research (2 048 tokens):

| Model ID | Name | Best for |
|---|---|---|
| `openai/gpt-oss-20b` | GPT OSS 20B ⭐ | Default — best speed/quality balance |
| `openai/gpt-oss-120b` | GPT OSS 120B | Most powerful reasoning |
| `qwen/qwen3.6-27b` | Qwen 3.6 27B | Most detailed, longest outputs |

The Critique Agent defaults to a **different model** than the Writer Agent to avoid self-preference
scoring bias. If a model is rate-limited, the pipeline automatically tries the next one.

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
| `HF_MODEL_ID` | optional | Default model for Research + Writer (default: `openai/gpt-oss-20b`) |
| `MAX_ITERATIONS` | optional | Max pipeline loops (default: 3) |
| `QUALITY_THRESHOLD` | optional | Score to stop iterating (default: 0.75) |
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

# Run with LLM faithfulness judge (slower, more accurate)
python -m src.evals.harness --faithfulness
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

## Planned Improvements

The following 14 improvements are scoped, prioritised, and tracked here. Items marked ✅ are complete.

### Tier 1 — Perception-changers

These change how the project is understood from the outside.

**1. Inline citations in reports** *(Tier 1 — in progress)*
The Writer Agent prompt currently does not require inline citations. Source URLs are retrieved from DuckDuckGo and present in the research notes, but the final report reads like an essay. Fix: update the Writer prompt to embed `[Source](url)` references after each claim and require at least one citation per Key Finding.

**2. Critique Agent checks source grounding** *(Tier 1 — in progress)*
The Critique Agent currently scores only writing quality (accuracy, completeness, clarity). It has no visibility into the retrieved sources. Fix: pass the retrieved source URLs into the Critique prompt and add a "are claims supported by retrieved text?" dimension to its rubric, making it a genuine RAG evaluator.

**3. Multi-rubric Critique scoring** *(Tier 1 — in progress)*
The Critique Agent returns a single composite score (0–1). A single number from the same model family that generated the report is weak evidence of quality. Fix: require the Critique Agent to return four separate sub-scores — `factual_support`, `coverage`, `structure`, `clarity` — each 0–1, plus an overall. Log all four in SQLite and surface them in the observability dashboard.

**4. Cross-model critique to reduce self-preference bias** *(Tier 1 — in progress)*
The Research, Writer, and Critique agents all default to the same model. A model grading its own output inflates scores. Fix: set the Critique Agent default to a *different* model than the Writer Agent (e.g., Writer=`gpt-oss-20b`, Critic=`qwen/qwen3.6-27b`). Expose this as a separate default in settings.

**5. Published eval results in the README** *(Tier 1 — in progress)*
The eval harness exists and runs 25 benchmark topics, but no results are committed to the repo. Without numbers, the harness is invisible to an interviewer. Fix: run the full harness across three configurations (single LLM call baseline, pipeline without critique loop, full pipeline), commit the JSON reports, and add a results table to this README comparing average overall score, citation accuracy, iteration count, latency, and cost.

### Tier 2 — Engineering quality

**6. Quality threshold calibrated to reality** *(Tier 2 — in progress)*
`QUALITY_THRESHOLD` defaults to `0.95`. In practice, the Critique Agent rarely scores above 0.85 on first pass, so the pipeline always runs to `MAX_ITERATIONS` and never actually converges early. This makes the self-correcting loop claim hollow. Fix: lower the default to `0.75` and align the `HF_MODEL_ID` setting default with the actual `DEFAULT_MODEL` in `llm.py` (currently mismatched).

**7. Research notes context cap** *(Tier 2 — in progress)*
`research_notes` uses `operator.add` to accumulate across all iterations. By iteration 3, the Writer receives 3× the notes, increasing LLM context usage and potentially causing overflow or degraded coherence. Fix: cap the notes fed to the Writer at the most recent 2 iterations (keep the full list in state for eval/audit, slice only what the Writer prompt receives).

**8. Per-user rate limiting on `/research/stream`** *(Tier 2 — in progress)*
There is no per-user throttle on the research endpoint. A single user can trigger unlimited concurrent pipeline runs, each making 9+ Groq API calls. Fix: add an in-memory `asyncio.Semaphore` per user ID, limiting each user to 1 concurrent run and returning HTTP 429 if they try to start a second before the first completes.

**9. Topic input sanitization** *(Tier 2 — in progress)*
Topics are passed directly from user input into DuckDuckGo queries and LLM prompts without length limits or sanitization. With live web retrieval, fetched page content creates a prompt injection surface. Fix: enforce a max topic length (e.g., 300 chars), strip control characters, and wrap fetched web content in a clearly delimited block in the prompt so injected instructions cannot override agent directives.

**10. GitHub Actions CI workflow** *(Tier 2 — in progress)*
There is no automated CI. The "CI/CD" skill listed on the resume is not validated by any workflow in this repo. Fix: add `.github/workflows/ci.yml` that runs on every push — installs dependencies, runs `ruff` linting, and executes a mocked-LLM smoke test against the graph routing logic and auth endpoints. This makes the CI/CD claim concrete and verifiable.

### Tier 3 — Polish

**11. README architecture diagram, demo GIF, and example output** *(Tier 3 — in progress)*
The README has a text-based ASCII diagram but no visual architecture diagram, no demo GIF, and no example input/output pair. Recruiters who skim repos in 30 seconds see none of the interesting parts. Fix: add a LangGraph flow diagram (exported from LangGraph's built-in `get_graph().draw_mermaid_png()`), a short screen-capture GIF of a live run, and an example topic with its full generated report excerpt.

**12. Model comparison table backed by data** *(Tier 3 — in progress)*
The README implies optimal cost/performance tradeoffs across models, but no comparison data exists. Fix: run the same 5 easy benchmark topics through all 3 available models, record average quality score and latency, and replace the current model table with actual measured values (e.g., avg score, avg latency, cost per run).

**13. Security hardening review** *(Tier 3 — in progress)*
Several low-hanging security issues: (a) `JWT_SECRET` has a hardcoded insecure default that ships as-is if `.env` is not configured, (b) CORS is fully open (`allow_origins=["*"]`), (c) API keys must never be logged — verify no key appears in `logger` calls. Fix: raise a startup error if `JWT_SECRET` equals the default value in production, narrow CORS origins via a `CORS_ORIGINS` env var, and audit all `logger.*` calls for credential leakage.

**14. Remove dead code and empty stubs** *(Tier 3 — in progress)*
Two vestigial items: (a) `src/mcp/` is an empty directory — the `mcp_tools` state field is populated but nothing reads it; either implement MCP tool integration or remove the stub, (b) `final_report` is declared in `ResearchState` and passed in `build_initial_state` but is never written by any agent. Fix: remove `final_report` from the state schema, delete the empty `src/mcp/` directory, and either implement MCP tool calling via LangGraph's `ToolNode` or explicitly document it as a future milestone.

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
**Fix:** Auto-rotation across 3 Groq models — if one is rate-limited the next is tried
immediately.

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
│   │   ├── research_agent.py    Web search + LLM extraction + source URLs
│   │   ├── writer_agent.py      Report generation with inline citations (4 or 9 sections)
│   │   └── critique_agent.py    4-rubric scoring with source grounding check
│   ├── graph/
│   │   └── research_graph.py    LangGraph StateGraph + conditional edges
│   ├── state/
│   │   └── research_state.py    ResearchState TypedDict
│   ├── tools/
│   │   ├── llm.py               Groq API — 3 models, auto-rotation
│   │   └── search_tool.py       DuckDuckGo web search (ddgs) with retry/backoff
│   ├── evals/
│   │   ├── dataset.py           25-topic benchmark with difficulty + gold facts
│   │   ├── metrics.py           Coverage, faithfulness, retrieval metrics
│   │   └── harness.py           CLI runner, versioned JSON reports
│   ├── observability/
│   │   └── tracing.py           LangSmith + Langfuse setup
│   ├── api.py                   FastAPI + SSE streaming + rate limiting + /observability
│   ├── auth.py                  JWT + bcrypt
│   └── database.py              SQLite: users, sessions, trace_events
├── frontend/
│   ├── index.html               Main research UI
│   ├── login.html               Auth page
│   └── observability.html       Trace timeline dashboard
├── .github/
│   └── workflows/
│       └── ci.yml               Lint + smoke tests on every push
├── .env.example
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · SQLite · python-jose · bcrypt

**LLM:** Groq API (free tier) — GPT OSS 20B/120B, Qwen 3.6 27B with cross-model critique

**Search:** DuckDuckGo (ddgs) with retry/backoff and instant-answer fallback

**Observability:** SQLite trace_events · LangSmith · Langfuse · built-in `/observability` page

**Eval:** 25-topic benchmark harness with deterministic + LLM-judge metrics

**Frontend:** Vanilla HTML · CSS · JavaScript (no framework)

**Deployment:** Docker · GitHub Actions CI
