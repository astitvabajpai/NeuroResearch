FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv + uvx (required for MCP servers) ───────────────────────────────
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Python dependencies (layer-cached) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────────────────────
COPY . .

# ── Create required directories ────────────────────────────────────────────────
RUN mkdir -p /data data/eval_results && chmod 777 /data

# ── Pre-download embedding model so first run isn't slow ──────────────────────
# Skip on HF Spaces (SPACE_ID is set) — RAG is disabled there anyway
ARG SPACE_ID=""
RUN if [ -z "$SPACE_ID" ]; then \
      python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'}); \
print('Embedding model cached')" || echo "Embedding pre-download skipped"; \
    fi

# ── Pre-install mcp-server-fetch via uvx so first MCP call is instant ─────────
RUN uvx mcp-server-fetch --help > /dev/null 2>&1 || echo "mcp-server-fetch pre-install skipped"

# ── HF Spaces persistent storage ──────────────────────────────────────────────
# HF Spaces mounts /data as persistent disk automatically
# database.py detects /data and uses it when writable

EXPOSE 7860

CMD ["python", "app.py"]
