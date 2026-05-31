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
        """)
        # Migrate existing DBs that predate the models column
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "models" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN models TEXT NOT NULL DEFAULT '{}'")


# ── User helpers ──────────────────────────────────────────────────────────────

def create_user(username: str, email: str, hashed_password: str) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?,?,?,?)",
            (username, email, hashed_password, datetime.utcnow().isoformat()),
        )
        return {"id": cur.lastrowid, "username": username, "email": email}


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
