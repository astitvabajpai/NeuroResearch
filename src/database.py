"""
SQLite database layer — users + research sessions.
Uses only the stdlib sqlite3 module, no ORM needed.
"""
import sqlite3
import os
import json
from datetime import datetime

# HF Spaces mounts persistent storage at /data; fall back to local data/ dir
_HF_DATA = "/data"
if os.path.isdir(_HF_DATA) and os.access(_HF_DATA, os.W_OK):
    DB_PATH = os.path.join(_HF_DATA, "app.db")
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.db")


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT    UNIQUE NOT NULL,
                email     TEXT    UNIQUE NOT NULL,
                password  TEXT    NOT NULL,
                created_at TEXT   NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT    PRIMARY KEY,
                user_id       INTEGER NOT NULL,
                topic         TEXT    NOT NULL,
                draft         TEXT    NOT NULL,
                critique_score REAL   NOT NULL DEFAULT 0,
                iterations    INTEGER NOT NULL DEFAULT 0,
                research_notes TEXT   NOT NULL DEFAULT '[]',
                models        TEXT    NOT NULL DEFAULT '{}',
                created_at    TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                node_name   TEXT    NOT NULL,
                iteration   INTEGER NOT NULL DEFAULT 0,
                input_keys  TEXT    NOT NULL DEFAULT '[]',
                output_keys TEXT    NOT NULL DEFAULT '[]',
                latency_ms  REAL    NOT NULL DEFAULT 0,
                error       TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        # Migrate existing DBs
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "models" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN models TEXT NOT NULL DEFAULT '{}'")


# ── User helpers ──────────────────────────────────────────────────────────────

def create_user(username: str, email: str, hashed_password: str) -> dict:
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, email, password, created_at) VALUES (?,?,?,?)",
                (username, email, hashed_password, datetime.utcnow().isoformat()),
            )
            return {"id": cur.lastrowid, "username": username, "email": email}
        except sqlite3.IntegrityError as e:
            msg = str(e)
            if "username" in msg:
                raise ValueError("Username already taken")
            raise ValueError("Email already registered")


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


# ── Session helpers ───────────────────────────────────────────────────────────

def save_session(session_id: str, user_id: int, topic: str, draft: str,
                 score: float, iterations: int, notes: list, models: str = "{}") -> dict:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, user_id, topic, draft, critique_score, iterations, research_notes, models, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id, user_id, topic, draft, score, iterations,
             json.dumps(notes), models, datetime.utcnow().isoformat()),
        )
    return {"id": session_id}


def get_sessions_for_user(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["research_notes"] = json.loads(d["research_notes"])
        result.append(d)
    return result


def get_session(session_id: str, user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["research_notes"] = json.loads(d["research_notes"])
    return d


def delete_session(session_id: str, user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        )
    return cur.rowcount > 0


# ── Trace event helpers ───────────────────────────────────────────────────────

def log_trace_event(
    session_id: str,
    node_name: str,
    iteration: int,
    input_keys: list,
    output_keys: list,
    latency_ms: float,
    error: str = "",
) -> None:
    """Record a single LangGraph node execution into trace_events."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO trace_events
               (session_id, node_name, iteration, input_keys, output_keys,
                latency_ms, error, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                session_id, node_name, iteration,
                json.dumps(input_keys), json.dumps(output_keys),
                latency_ms, error,
                datetime.utcnow().isoformat(),
            ),
        )


def get_trace_events(session_id: str) -> list[dict]:
    """Return all trace events for a session, ordered by creation time."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["input_keys"]  = json.loads(d["input_keys"])
        d["output_keys"] = json.loads(d["output_keys"])
        result.append(d)
    return result


def get_all_trace_sessions(limit: int = 100) -> list[dict]:
    """Return recent sessions that have trace events, with summary stats."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.id, s.topic, s.critique_score, s.iterations, s.created_at,
                      COUNT(t.id) AS event_count,
                      SUM(t.latency_ms) AS total_latency_ms,
                      SUM(CASE WHEN t.error != '' THEN 1 ELSE 0 END) AS error_count
               FROM sessions s
               LEFT JOIN trace_events t ON s.id = t.session_id
               GROUP BY s.id
               ORDER BY s.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
