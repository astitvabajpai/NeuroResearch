# 🧠 NeuroResearch — Self-Correcting Multi-Agent Research System

A production-ready AI research pipeline built with **LangGraph**, **FastAPI**, and **HuggingFace Inference API**. Three specialized agents collaborate in a self-correcting loop to produce high-quality research reports on any topic.

🔗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/johncenaqweewewwe/neuroResearch)
🔗 **GitHub:** [astitvabajpai/NeuroResearch](https://github.com/astitvabajpai/NeuroResearch)

---

## How It Works

```
Topic → Research Agent → Writer Agent → Critique Agent
                ↑                              |
                └──── self-correct if score < 0.8 ────┘
```

| Agent | Role |
|---|---|
| 🔍 **Research Agent** | Searches the web via DuckDuckGo, extracts key findings |
| ✍️ **Writer Agent** | Synthesizes notes into a structured markdown report |
| 🧐 **Critique Agent** | Scores the draft (0.0–1.0) on accuracy, completeness, clarity |

The pipeline iterates automatically until the quality score hits **0.8** or max iterations are reached.

---

## Research Modes

| | Normal Mode | 🔬 Deep Research Mode |
|---|---|---|
| Web searches | 1 | 2 (topic + latest advances) |
| Report sections | 4 | 9 (Executive Summary → Future Directions) |
| Min word count | 400 | 800+ |
| Max tokens | 1024 | 2048 |
| Critique depth | Basic | Academic-level |

---

## Features

- 🔐 **JWT Authentication** — register/sign in, all sessions are private per user
- 🤖 **Per-agent model selection** — choose different models for each agent
- 🔬 **Deep Research mode** — comprehensive 9-section reports with 800+ words
- 📡 **Live SSE streaming** — watch each agent work in real time
- 📂 **Session history** — all runs saved, searchable, reloadable from sidebar
- 📄 **PDF export** — structured print-ready report with cover page and metadata
- 🌙 **Dark UI** — polished interface built with vanilla HTML/CSS/JS

---

## Available Models

| Model | Tag |
|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⭐ Recommended |
| `Qwen/Qwen2.5-7B-Instruct` | Strong |
| `mistralai/Mistral-7B-Instruct-v0.3` | Fast |
| `microsoft/Phi-3-mini-4k-instruct` | Lightweight |
| `HuggingFaceH4/zephyr-7b-beta` | — |
| `google/gemma-2-2b-it` | Small |
| `tiiuae/falcon-7b-instruct` | — |

> **Note:** Llama 3 requires [requesting access](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on HuggingFace first.

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/astitvabajpai/NeuroResearch
cd NeuroResearch

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your HF_API_TOKEN and JWT_SECRET

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — you'll land on the login page.

---

## Deploying to HuggingFace Spaces

1. Create a new Space → **Docker SDK** → **Blank template**
2. Push this repo to the Space's git remote
3. Add these secrets in **Space → Settings → Variables and secrets**:

| Variable | Required | Description |
|---|---|---|
| `HF_API_TOKEN` | ✅ | Your HuggingFace token (read access) |
| `JWT_SECRET` | ✅ | Any random secret string |
| `HF_MODEL_ID` | optional | Default: `meta-llama/Meta-Llama-3-8B-Instruct` |

4. Enable **Persistent Storage** (Space → Settings) so the SQLite database survives restarts

---

## Project Structure

```
├── src/
│   ├── agents/          # Research, Writer, Critique agents
│   ├── graph/           # LangGraph pipeline
│   ├── state/           # ResearchState TypedDict
│   ├── tools/           # HF LLM wrapper, search tool
│   ├── api.py           # FastAPI app + SSE streaming
│   ├── auth.py          # JWT + bcrypt auth
│   └── database.py      # SQLite layer
├── frontend/
│   ├── index.html       # Main app UI
│   └── login.html       # Auth page
├── Dockerfile
├── requirements.txt
└── app.py               # HF Spaces entrypoint (port 7860)
```

---

## Tech Stack

**Backend:** FastAPI · LangGraph · LangChain · SQLite · JWT (python-jose) · passlib/bcrypt

**LLM:** HuggingFace Inference API (serverless, free tier) via `router.huggingface.co`

**Search:** DuckDuckGo Search with retry + fallback

**Frontend:** Vanilla HTML · CSS · JavaScript (no framework)

**Deployment:** Docker on HuggingFace Spaces

---

## Contributing

Fork the repo, make your changes, and open a PR. Feature ideas welcome in Issues!
