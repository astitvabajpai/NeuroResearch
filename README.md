---
title: NeuroResearch — Self-Correcting Multi-Agent Research System
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# NeuroResearch — Self-Correcting Multi-Agent Research System

A production-ready AI research pipeline built with **LangGraph**, **FastAPI**, and **HuggingFace Inference API**.

## How It Works

Three specialized agents collaborate in a self-correcting loop:

| Agent | Role |
|---|---|
| 🔍 **Research Agent** | Searches the web via DuckDuckGo and extracts key findings |
| ✍️ **Writer Agent** | Synthesizes notes into a structured report |
| 🧐 **Critique Agent** | Scores the draft (0–1). If below threshold, the loop repeats |

The pipeline runs up to N iterations until the quality score hits **0.8** or max iterations are reached.

## Features

- 🔐 JWT authentication — register / sign in, sessions are private per user
- 🤖 Per-agent model selection — choose from 7 free HuggingFace models per agent
- 📡 Live SSE streaming — watch each agent work in real time
- 📂 Session history — all research runs saved, searchable, reloadable
- 📄 PDF download — structured, print-ready report with cover page
- 🌙 Dark UI — polished interface matching the Pencil design system

## Environment Variables

Set these in HuggingFace Spaces → Settings → Repository secrets:

| Variable | Required | Description |
|---|---|---|
| `HF_API_TOKEN` | ✅ | Your HuggingFace API token (read access) |
| `JWT_SECRET` | ✅ | Random secret string for signing auth tokens |
| `HF_MODEL_ID` | optional | Default model (default: `meta-llama/Meta-Llama-3-8B-Instruct`) |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in env vars
cp .env.example .env

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain, SQLite
- **LLM**: HuggingFace Inference API (serverless, free tier)
- **Auth**: JWT + bcrypt
- **Frontend**: Vanilla HTML/CSS/JS (no framework)
- **Deployment**: Docker on HuggingFace Spaces
