import asyncio
import json
import os
import uuid
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

app = FastAPI(title="NeuroResearch — Multi-Agent Research System")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Resolve frontend path relative to this file, works both locally and in Docker
_BASE_DIR = Path(__file__).resolve().parent.parent
frontend_path = str(_BASE_DIR / "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.on_event("startup")
def startup():
    init_db()


def _read(name: str) -> str:
    with open(Path(frontend_path) / name, "r", encoding="utf-8") as f:
        return f.read()


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_app():   return HTMLResponse(_read("index.html"))

@app.get("/login", response_class=HTMLResponse)
async def serve_login(): return HTMLResponse(_read("login.html"))

@app.get("/health")
async def health():      return {"status": "ok"}


# ── Models endpoint ───────────────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    from src.tools.hg_llm import AVAILABLE_MODELS
    return AVAILABLE_MODELS


# ── Auth ──────────────────────────────────────────────────────────────────────

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
    user = create_user(req.username.strip(), req.email.lower(), hash_password(req.password))
    return {"token": create_token(user["id"]), "user": {"id": user["id"], "username": user["username"], "email": user["email"]}}


@app.post("/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_email(req.email.lower())
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid email or password")
    return {"token": create_token(user["id"]), "user": {"id": user["id"], "username": user["username"], "email": user["email"]}}


@app.get("/auth/me")
async def me(current_user=Depends(get_current_user)):
    return {"id": current_user["id"], "username": current_user["username"], "email": current_user["email"]}


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions(current_user=Depends(get_current_user)):
    sessions = get_sessions_for_user(current_user["id"])
    return [
        {
            "id": s["id"], "topic": s["topic"],
            "critique_score": s["critique_score"], "iterations": s["iterations"],
            "created_at": s["created_at"],
            "preview": s["draft"][:120] + "..." if len(s["draft"]) > 120 else s["draft"],
            "models": json.loads(s.get("models", "{}") or "{}"),
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


# ── PDF download for a saved session ─────────────────────────────────────────

@app.get("/sessions/{session_id}/pdf")
async def download_pdf(session_id: str, t: str = Query(default=None)):
    """PDF download — accepts token via ?t= query param so it works in a new browser tab."""
    from src.auth import decode_token
    from src.database import get_user_by_id
    if not t:
        raise HTTPException(401, "Token required")
    try:
        user_id = decode_token(t)
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(401, "Invalid token")
    except Exception:
        raise HTTPException(401, "Invalid token")
    s = get_session(session_id, user_id)
    if not s:
        raise HTTPException(404, "Session not found")
    html = _build_pdf_html(s["topic"], s["draft"], s["critique_score"],
                           s["iterations"], s["created_at"])
    return HTMLResponse(content=html)


def _build_pdf_html(topic: str, draft: str, score: float, iterations: int, created_at: str) -> str:
    """Return a print-ready HTML page that auto-opens the print dialog."""
    score_pct = f"{score * 100:.0f}%"
    score_color = "#10B981" if score >= 0.8 else "#F59E0B" if score >= 0.5 else "#EF4444"
    try:
        dt = datetime.fromisoformat(created_at).strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        dt = created_at

    # Convert markdown to basic HTML
    body_html = _md_to_html(draft)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{topic}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', Georgia, serif; max-width: 820px; margin: 0 auto; padding: 48px 56px; color: #111; line-height: 1.75; font-size: 15px; }}
    .cover {{ border-bottom: 3px solid #582CFF; padding-bottom: 28px; margin-bottom: 36px; }}
    .cover-tag {{ font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #582CFF; margin-bottom: 12px; }}
    .cover-title {{ font-size: 26px; font-weight: 700; line-height: 1.3; color: #0a0a0f; margin-bottom: 16px; }}
    .cover-meta {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #555; }}
    .cover-meta span {{ display: flex; align-items: center; gap: 6px; }}
    .score-badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; background: {score_color}18; color: {score_color}; border: 1px solid {score_color}44; }}
    h2 {{ font-size: 18px; font-weight: 700; color: #0a0a0f; margin: 32px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e5e7eb; }}
    h3 {{ font-size: 15px; font-weight: 600; color: #1a1a2e; margin: 22px 0 8px; }}
    p  {{ margin-bottom: 14px; color: #333; }}
    ul, ol {{ padding-left: 22px; margin-bottom: 14px; }}
    li {{ margin-bottom: 6px; color: #333; }}
    strong {{ font-weight: 600; color: #111; }}
    em {{ font-style: italic; color: #444; }}
    hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 28px 0; }}
    .footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #999; display: flex; justify-content: space-between; }}
    @media print {{
      body {{ padding: 24px 32px; }}
      @page {{ margin: 20mm; }}
    }}
  </style>
</head>
<body>
  <div class="cover">
    <div class="cover-tag">NeuroResearch — AI Research Report</div>
    <div class="cover-title">{topic}</div>
    <div class="cover-meta">
      <span>📅 {dt}</span>
      <span>🔄 {iterations} iteration{"s" if iterations != 1 else ""}</span>
      <span>Quality: <span class="score-badge">{score_pct}</span></span>
    </div>
  </div>

  {body_html}

  <div class="footer">
    <span>Generated by NeuroResearch Multi-Agent System</span>
    <span>{dt}</span>
  </div>

  <script>window.onload = () => {{ window.print(); }}</script>
</body>
</html>"""


def _md_to_html(text: str) -> str:
    import re, html as html_lib
    t = html_lib.escape(text)
    t = re.sub(r'^### (.+)$', r'<h3>\1</h3>', t, flags=re.MULTILINE)
    t = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', t, flags=re.MULTILINE)
    t = re.sub(r'^# (.+)$',   r'<h2>\1</h2>', t, flags=re.MULTILINE)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', t)
    t = re.sub(r'^---$', '<hr>', t, flags=re.MULTILINE)
    t = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', t, flags=re.MULTILINE)
    t = re.sub(r'^[-•]\s+(.+)$',  r'<li>\1</li>', t, flags=re.MULTILINE)
    t = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group()}</ul>', t, flags=re.DOTALL)
    # Wrap bare paragraphs
    lines = t.split('\n')
    out, buf = [], []
    for line in lines:
        if line.strip() == '':
            if buf:
                out.append('<p>' + ' '.join(buf) + '</p>')
                buf = []
        elif line.startswith('<'):
            if buf:
                out.append('<p>' + ' '.join(buf) + '</p>')
                buf = []
            out.append(line)
        else:
            buf.append(line)
    if buf:
        out.append('<p>' + ' '.join(buf) + '</p>')
    return '\n'.join(out)


# ── Research streaming ────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    max_iterations: int = 3
    research_model: Optional[str] = None
    writer_model:   Optional[str] = None
    critique_model: Optional[str] = None
    deep_research:  bool = False


async def run_research_stream(
    topic: str, max_iterations: int, user_id: int,
    research_model: str, writer_model: str, critique_model: str,
    deep_research: bool = False,
) -> AsyncGenerator[str, None]:

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield send("status", {"message": "Initializing research pipeline...", "phase": "init"})

    try:
        from src.graph.research_graph import get_research_app
        loop = asyncio.get_event_loop()

        yield send("status", {"message": "Building agent graph...", "phase": "init"})
        research_app = await loop.run_in_executor(None, get_research_app)
        yield send("status", {"message": f'Starting research on: "{topic}"', "phase": "start"})

        from src.tools.hg_llm import DEFAULT_MODEL
        initial_state = {
            "topic": topic,
            "research_notes": [],
            "draft": "",
            "critique_feedback": "",
            "critique_score": 0.0,
            "iteration": 0,
            "max_iterations": max_iterations,
            "final_report": "",
            "research_model": research_model or DEFAULT_MODEL,
            "writer_model":   writer_model   or DEFAULT_MODEL,
            "critique_model": critique_model or DEFAULT_MODEL,
            "deep_research":  deep_research,
        }

        iteration = 0
        final_state = dict(initial_state)  # track accumulated state

        async for chunk in research_app.astream(initial_state):
            for node_name, node_output in chunk.items():
                # Merge node output into final_state
                for k, v in node_output.items():
                    if k == "research_notes" and isinstance(v, list):
                        final_state["research_notes"] = final_state.get("research_notes", []) + v
                    else:
                        final_state[k] = v

                if node_name == "research":
                    iteration += 1
                    notes = node_output.get("research_notes", [])
                    latest = notes[-1] if notes else ""
                    yield send("agent_update", {
                        "agent": "Research Agent", "icon": "🔍", "iteration": iteration,
                        "message": f"Iteration {iteration}: Gathered research notes",
                        "detail": latest[:500] + ("..." if len(latest) > 500 else ""),
                        "phase": "research",
                    })
                elif node_name == "write":
                    draft = node_output.get("draft", "")
                    if hasattr(draft, "content"): draft = draft.content
                    final_state["draft"] = draft
                    yield send("agent_update", {
                        "agent": "Writer Agent", "icon": "✍️", "iteration": iteration,
                        "message": f"Iteration {iteration}: Draft written",
                        "detail": draft[:500] + ("..." if len(draft) > 500 else ""),
                        "phase": "write",
                    })
                elif node_name == "critique":
                    score    = node_output.get("critique_score", 0.0)
                    feedback = node_output.get("critique_feedback", "")
                    yield send("agent_update", {
                        "agent": "Critique Agent", "icon": "🧐", "iteration": iteration,
                        "message": f"Iteration {iteration}: Quality score {score:.2f}",
                        "detail": feedback[:500] + ("..." if len(feedback) > 500 else ""),
                        "score": score, "phase": "critique",
                    })

        # Use accumulated state — no second invoke needed
        yield send("status", {"message": "Saving report...", "phase": "finalizing"})

        draft = final_state.get("draft", "")
        if hasattr(draft, "content"): draft = draft.content

        session_id = str(uuid.uuid4())
        models_used = json.dumps({
            "research": initial_state["research_model"],
            "writer":   initial_state["writer_model"],
            "critique": initial_state["critique_model"],
        })
        save_session(
            session_id, user_id, topic, draft,
            final_state.get("critique_score", 0.0),
            final_state.get("iteration", iteration),
            final_state.get("research_notes", []),
            models_used,
        )

        yield send("complete", {
            "session_id": session_id, "topic": topic, "draft": draft,
            "critique_score": final_state.get("critique_score", 0.0),
            "critique_feedback": final_state.get("critique_feedback", ""),
            "iterations": final_state.get("iteration", iteration),
            "research_notes": final_state.get("research_notes", []),
            "deep_research": deep_research,
            "models": {
                "research": initial_state["research_model"],
                "writer":   initial_state["writer_model"],
                "critique": initial_state["critique_model"],
            },
        })

    except Exception as e:
        yield send("error", {"message": str(e)})


@app.post("/research/stream")
async def research_stream(req: ResearchRequest, current_user=Depends(get_current_user)):
    return StreamingResponse(
        run_research_stream(
            req.topic, req.max_iterations, current_user["id"],
            req.research_model, req.writer_model, req.critique_model,
            req.deep_research,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
