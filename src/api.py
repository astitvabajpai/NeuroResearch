"""
FastAPI application — HTTP endpoints + SSE streaming research pipeline.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.auth import create_token, get_current_user, hash_password, verify_password
from src.database import (
    create_user, delete_session, get_session,
    get_sessions_for_user, get_user_by_email, init_db, save_session,
)

# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        from src.observability.tracing import init_observability
        init_observability()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Observability init failed: %s", exc)
    yield

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroResearch — Multi-Agent Research System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve frontend path relative to this file — works locally and in Docker
_BASE_DIR     = Path(__file__).resolve().parent.parent
frontend_path = str(_BASE_DIR / "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read(name: str) -> str:
    with open(Path(frontend_path) / name, "r", encoding="utf-8") as f:
        return f.read()


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_app():
    return HTMLResponse(_read("index.html"))

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    return HTMLResponse(_read("login.html"))

@app.get("/observability", response_class=HTMLResponse)
async def serve_observability():
    return HTMLResponse(_read("observability.html"))

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Models ─────────────────────────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    from src.tools.llm import AVAILABLE_MODELS
    return AVAILABLE_MODELS


# ── Auth ───────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest):
    if len(req.username.strip()) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if get_user_by_email(req.email):
        raise HTTPException(400, "Email already registered")
    try:
        user = create_user(
            req.username.strip(),
            req.email.lower(),
            hash_password(req.password),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "token": create_token(user["id"]),
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
    }


@app.post("/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_email(req.email.lower())
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid email or password")
    return {
        "token": create_token(user["id"]),
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
    }


@app.get("/auth/me")
async def me(current_user=Depends(get_current_user)):
    return {
        "id":       current_user["id"],
        "username": current_user["username"],
        "email":    current_user["email"],
    }


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions(current_user=Depends(get_current_user)):
    sessions = get_sessions_for_user(current_user["id"])
    return [
        {
            "id":             s["id"],
            "topic":          s["topic"],
            "critique_score": s["critique_score"],
            "iterations":     s["iterations"],
            "created_at":     s["created_at"],
            "preview":        s["draft"][:120] + "…" if len(s["draft"]) > 120 else s["draft"],
            "models":         json.loads(s.get("models", "{}") or "{}"),
        }
        for s in sessions
    ]


@app.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, current_user=Depends(get_current_user)):
    s = get_session(session_id, current_user["id"])
    if not s:
        raise HTTPException(404, "Session not found")
    s["models"] = json.loads(s.get("models", "{}") or "{}")
    return s


@app.delete("/sessions/{session_id}", status_code=204)
async def remove_session(session_id: str, current_user=Depends(get_current_user)):
    if not delete_session(session_id, current_user["id"]):
        raise HTTPException(404, "Session not found")


# ── Traces (observability) ─────────────────────────────────────────────────────

@app.get("/traces")
async def list_traces(current_user=Depends(get_current_user)):
    """Return all sessions with trace summary stats."""
    from src.database import get_all_trace_sessions
    return get_all_trace_sessions(limit=100)


@app.get("/traces/{session_id}")
async def get_trace(session_id: str, current_user=Depends(get_current_user)):
    """Return per-node trace events for a session."""
    from src.database import get_trace_events
    s = get_session(session_id, current_user["id"])
    if not s:
        raise HTTPException(404, "Session not found")
    events = get_trace_events(session_id)
    return {"session_id": session_id, "topic": s["topic"], "events": events}


# ── PDF ────────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/pdf")
async def download_pdf(session_id: str, t: str = Query(default=None)):
    """Print-ready HTML — accepts ?t= token so it works in a new browser tab."""
    from src.auth import decode_token
    from src.database import get_user_by_id
    if not t:
        raise HTTPException(401, "Token required")
    try:
        user_id = decode_token(t)
        user    = get_user_by_id(user_id)
        if not user:
            raise HTTPException(401, "Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Invalid token")
    s = get_session(session_id, user_id)
    if not s:
        raise HTTPException(404, "Session not found")
    html = _build_pdf_html(
        s["topic"], s["draft"],
        s["critique_score"], s["iterations"], s["created_at"],
    )
    return HTMLResponse(content=html)


def _build_pdf_html(
    topic: str, draft: str, score: float, iterations: int, created_at: str
) -> str:
    score_pct   = f"{score * 100:.0f}%"
    score_color = "#10B981" if score >= 0.8 else "#F59E0B" if score >= 0.5 else "#EF4444"
    try:
        dt = datetime.fromisoformat(created_at).strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        dt = created_at
    body_html = _md_to_html(draft)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{topic}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Inter',Georgia,serif;max-width:820px;margin:0 auto;padding:48px 56px;color:#111;line-height:1.75;font-size:15px}}
    .cover{{border-bottom:3px solid #582CFF;padding-bottom:28px;margin-bottom:36px}}
    .cover-tag{{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#582CFF;margin-bottom:12px}}
    .cover-title{{font-size:26px;font-weight:700;line-height:1.3;color:#0a0a0f;margin-bottom:16px}}
    .cover-meta{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:#555}}
    .score-badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:700;background:{score_color}18;color:{score_color};border:1px solid {score_color}44}}
    h2{{font-size:18px;font-weight:700;color:#0a0a0f;margin:32px 0 10px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}}
    h3{{font-size:15px;font-weight:600;color:#1a1a2e;margin:22px 0 8px}}
    p{{margin-bottom:14px;color:#333}}
    ul,ol{{padding-left:22px;margin-bottom:14px}}
    li{{margin-bottom:6px;color:#333}}
    strong{{font-weight:600;color:#111}}
    em{{font-style:italic;color:#444}}
    hr{{border:none;border-top:1px solid #e5e7eb;margin:28px 0}}
    code{{font-family:monospace;font-size:13px;background:#f3f4f6;padding:1px 5px;border-radius:4px}}
    .footer{{margin-top:48px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:12px;color:#999;display:flex;justify-content:space-between}}
    @media print{{body{{padding:24px 32px}}@page{{margin:20mm}}}}
  </style>
</head>
<body>
  <div class="cover">
    <div class="cover-tag">NeuroResearch - AI Research Report</div>
    <div class="cover-title">{topic}</div>
    <div class="cover-meta">
      <span>Date: {dt}</span>
      <span>Iterations: {iterations}</span>
      <span>Quality: <span class="score-badge">{score_pct}</span></span>
    </div>
  </div>
  {body_html}
  <div class="footer">
    <span>Generated by NeuroResearch Multi-Agent System</span>
    <span>{dt}</span>
  </div>
  <script>window.onload=()=>window.print()</script>
</body>
</html>"""


def _md_to_html(text: str) -> str:
    import re
    import html as html_lib
    t     = html_lib.escape(text)
    lines = t.split("\n")
    out: list = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if re.match(r"^### ", s):
            out.append(f"<h3>{_inline(s[4:])}</h3>"); i += 1; continue
        if re.match(r"^## ", s):
            out.append(f"<h2>{_inline(s[3:])}</h2>"); i += 1; continue
        if re.match(r"^# ", s):
            out.append(f"<h2>{_inline(s[2:])}</h2>"); i += 1; continue
        if re.match(r"^---+$", s):
            out.append("<hr>"); i += 1; continue
        if re.match(r"^\d+\.\s+", s):
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                out.append(f"<li>{_inline(re.sub(r'^[0-9]+[.] ', '', lines[i].strip()))}</li>")
                i += 1
            out.append("</ol>"); continue
        if re.match(r"^[-*]\s+", s):
            out.append("<ul>")
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                out.append(f"<li>{_inline(re.sub(r'^[-*] ', '', lines[i].strip()))}</li>")
                i += 1
            out.append("</ul>"); continue
        if s == "":
            out.append("<br>"); i += 1; continue
        out.append(f"<p>{_inline(s)}</p>"); i += 1
    return "\n".join(out)


def _inline(text: str) -> str:
    import re
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>",          text)
    text = re.sub(r"\*(.+?)\*",         r"<em>\1</em>",                  text)
    text = re.sub(r"`(.+?)`",           r"<code>\1</code>",              text)
    return text


# ── Research streaming ─────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic:          str
    max_iterations: int           = 3
    research_model: Optional[str] = None
    writer_model:   Optional[str] = None
    critique_model: Optional[str] = None
    deep_research:  bool          = False


async def run_research_stream(
    topic: str,
    max_iterations: int,
    user_id: int,
    research_model: Optional[str],
    writer_model: Optional[str],
    critique_model: Optional[str],
    deep_research: bool = False,
) -> AsyncGenerator[str, None]:

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield send("status", {"message": "Initializing research pipeline...", "phase": "init"})

    try:
        from src.graph.research_graph import build_initial_state, get_research_app
        from src.observability.tracing import build_langgraph_config, score_langfuse_trace
        from src.tools.llm import DEFAULT_MODEL, DEFAULT_WRITER_MODEL, DEFAULT_CRITIC_MODEL

        session_id = str(uuid.uuid4())
        loop       = asyncio.get_event_loop()

        yield send("status", {"message": "Building agent graph...", "phase": "init"})
        research_app = await loop.run_in_executor(None, get_research_app)

        res_model = research_model or DEFAULT_MODEL
        wrt_model = writer_model   or DEFAULT_WRITER_MODEL
        crt_model = critique_model or DEFAULT_CRITIC_MODEL

        lg_config, trace_id = build_langgraph_config(
            topic=topic,
            user_id=user_id,
            session_id=session_id,
            deep_research=deep_research,
        )

        initial_state = build_initial_state(
            topic=topic,
            max_iterations=max_iterations,
            research_model=res_model,
            writer_model=wrt_model,
            critique_model=crt_model,
            deep_research=deep_research,
            trace_id=trace_id,
        )

        yield send("status", {
            "message": f'Starting research on: "{topic}"',
            "phase": "start",
        })

        iteration   = 0
        final_state = dict(initial_state)
        # per-node timing: record wall time at the START of each chunk
        _node_start_times: dict = {}

        async for chunk in research_app.astream(initial_state, config=lg_config):
            for node_name, node_output in chunk.items():
                node_wall_start = time.time()
                for k, v in node_output.items():
                    if k == "research_notes" and isinstance(v, list):
                        final_state["research_notes"] = (
                            final_state.get("research_notes", []) + v
                        )
                    else:
                        final_state[k] = v
                node_latency_ms = (time.time() - node_wall_start) * 1000

                if node_name == "research":
                    iteration += 1
                    notes  = node_output.get("research_notes", [])
                    latest = notes[-1] if notes else ""
                    yield send("agent_update", {
                        "agent":      "Research Agent",
                        "icon":       "🔍",
                        "iteration":  iteration,
                        "message":    f"Iteration {iteration}: Gathered research notes",
                        "detail":     latest[:500] + ("..." if len(latest) > 500 else ""),
                        "phase":      "research",
                        "latency_ms": round(node_latency_ms),
                    })

                elif node_name == "write":
                    draft = node_output.get("draft", "")
                    if hasattr(draft, "content"):
                        draft = draft.content
                    final_state["draft"] = draft
                    yield send("agent_update", {
                        "agent":      "Writer Agent",
                        "icon":       "✍️",
                        "iteration":  iteration,
                        "message":    f"Iteration {iteration}: Draft written",
                        "detail":     draft[:500] + ("..." if len(draft) > 500 else ""),
                        "phase":      "write",
                        "latency_ms": round(node_latency_ms),
                    })

                elif node_name == "critique":
                    score    = node_output.get("critique_score",    0.0)
                    feedback = node_output.get("critique_feedback", "")
                    yield send("agent_update", {
                        "agent":      "Critique Agent",
                        "icon":       "🧐",
                        "iteration":  iteration,
                        "message":    f"Iteration {iteration}: Quality score {score:.2f}",
                        "detail":     feedback[:500] + ("..." if len(feedback) > 500 else ""),
                        "score":      score,
                        "phase":      "critique",
                        "latency_ms": round(node_latency_ms),
                    })

                # Store latency for trace logging after session is saved
                _node_start_times[f"{node_name}_{iteration}"] = node_latency_ms

        yield send("status", {"message": "Saving report...", "phase": "finalizing"})

        draft = final_state.get("draft", "")
        if hasattr(draft, "content"):
            draft = draft.content

        final_score    = float(final_state.get("critique_score", 0.0))
        final_iter     = int(final_state.get("iteration", iteration))
        final_notes    = final_state.get("research_notes", [])
        final_feedback = final_state.get("critique_feedback", "")

        models_used = json.dumps({
            "research": res_model,
            "writer":   wrt_model,
            "critique": crt_model,
        })
        save_session(
            session_id, user_id, topic,
            draft, final_score, final_iter, final_notes, models_used,
        )

        # Log trace events AFTER session is saved (FK constraint)
        try:
            from src.database import log_trace_event
            node_map = {
                "research": (["topic", "research_model"], ["research_notes", "iteration"]),
                "write":    (["topic", "research_notes", "writer_model"], ["draft"]),
                "critique": (["topic", "draft", "critique_model"], ["critique_score", "critique_feedback"]),
            }
            for key, latency_ms in _node_start_times.items():
                parts = key.rsplit("_", 1)
                node_name = parts[0]
                iter_num  = int(parts[1]) if len(parts) > 1 else 0
                in_keys, out_keys = node_map.get(node_name, ([], []))
                log_trace_event(session_id, node_name, iter_num,
                                in_keys, out_keys, latency_ms)
        except Exception:
            pass

        try:
            score_langfuse_trace(trace_id=trace_id, score=final_score, comment=final_feedback)
        except Exception:
            pass

        yield send("complete", {
            "session_id":        session_id,
            "topic":             topic,
            "draft":             draft,
            "critique_score":    final_score,
            "critique_feedback": final_feedback,
            "iterations":        final_iter,
            "research_notes":    final_notes,
            "deep_research":     deep_research,
            "trace_id":          trace_id,
            "models": {
                "research": res_model,
                "writer":   wrt_model,
                "critique": crt_model,
            },
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        yield send("error", {"message": str(exc)})


@app.post("/research/stream")
async def research_stream(
    req: ResearchRequest,
    current_user=Depends(get_current_user),
):
    return StreamingResponse(
        run_research_stream(
            req.topic,
            req.max_iterations,
            current_user["id"],
            req.research_model,
            req.writer_model,
            req.critique_model,
            req.deep_research,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
