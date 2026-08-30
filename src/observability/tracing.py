"""
Observability layer — LangSmith + Langfuse integration.

Both integrations are fully optional:
- LangSmith  : activated by setting env vars; auto-instruments all LangChain/
               LangGraph calls at zero code cost.
- Langfuse   : uses an explicit CallbackHandler injected into the LangGraph
               run config, and scores the trace after each run.

Usage
-----
    config, trace_id = build_langgraph_config(
        topic=topic, user_id=user_id, session_id=session_id
    )
    async for chunk in research_app.astream(initial_state, config=config):
        ...
    score_langfuse_trace(trace_id, score=final_score, comment=feedback)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── LangSmith ─────────────────────────────────────────────────────────────────

def configure_langsmith() -> bool:
    """
    Set env vars so LangChain's auto-instrumentation activates.
    Returns True if LangSmith tracing is now enabled.
    """
    import os
    try:
        from src.config.settings import get_settings
        s = get_settings()
        if not s.LANGCHAIN_TRACING_V2 or not s.LANGCHAIN_API_KEY:
            logger.info("[Observability] LangSmith disabled (LANGCHAIN_TRACING_V2 or LANGCHAIN_API_KEY not set).")
            return False

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY",     s.LANGCHAIN_API_KEY)
        os.environ.setdefault("LANGCHAIN_PROJECT",     s.LANGCHAIN_PROJECT)
        os.environ.setdefault("LANGCHAIN_ENDPOINT",    s.LANGCHAIN_ENDPOINT)
        logger.info("[Observability] LangSmith tracing enabled (project=%s).", s.LANGCHAIN_PROJECT)
        return True
    except Exception as exc:
        logger.warning("[Observability] LangSmith setup failed: %s", exc)
        return False


# ── Langfuse ──────────────────────────────────────────────────────────────────

_langfuse_client: Any = None


def get_langfuse() -> Any:
    """Return the Langfuse client singleton, or None if not configured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from src.config.settings import get_settings
        s = get_settings()
        if not s.LANGFUSE_PUBLIC_KEY or not s.LANGFUSE_SECRET_KEY:
            logger.info("[Observability] Langfuse disabled (keys not set).")
            return None
        from langfuse import Langfuse  # type: ignore[import]
        _langfuse_client = Langfuse(
            public_key=s.LANGFUSE_PUBLIC_KEY,
            secret_key=s.LANGFUSE_SECRET_KEY,
            host=s.LANGFUSE_HOST,
        )
        logger.info("[Observability] Langfuse connected (host=%s).", s.LANGFUSE_HOST)
        return _langfuse_client
    except ImportError:
        logger.info("[Observability] langfuse package not installed — Langfuse disabled.")
        return None
    except Exception as exc:
        logger.warning("[Observability] Langfuse init failed: %s", exc)
        return None


def _make_langfuse_callback(
    trace_id: str,
    session_id: str,
    user_id: str,
    topic: str,
) -> Any:
    """
    Build a Langfuse CallbackHandler.  Compatible with both langfuse 2.x and 3.x
    by trying different import paths gracefully.
    Returns None if Langfuse is not configured or the callback cannot be built.
    """
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        # langfuse >= 2.x
        from langfuse.callback import CallbackHandler  # type: ignore[import]
    except ImportError:
        try:
            # older path
            from langfuse.langchain import CallbackHandler  # type: ignore[import]
        except ImportError:
            logger.info("[Observability] Langfuse CallbackHandler not available.")
            return None

    try:
        return CallbackHandler(
            public_key=lf.public_key,
            secret_key=lf.secret_key,
            host=lf.host,
            trace_id=trace_id,
            session_id=session_id,
            user_id=str(user_id),
            metadata={"topic": topic},
        )
    except Exception as exc:
        logger.warning("[Observability] Could not create Langfuse callback: %s", exc)
        return None


def score_langfuse_trace(
    trace_id: str,
    score: float,
    name: str = "critique_score",
    comment: str = "",
) -> None:
    """Push a numeric quality score to an existing Langfuse trace."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.score(trace_id=trace_id, name=name, value=score, comment=comment)
        logger.debug("[Observability] Scored trace %s: %s=%.3f", trace_id, name, score)
    except Exception as exc:
        logger.warning("[Observability] Langfuse scoring failed: %s", exc)


# ── LangGraph run config builder ──────────────────────────────────────────────

def build_langgraph_config(
    topic: str,
    user_id: int,
    session_id: str,
    deep_research: bool = False,
    extra_tags: Optional[List[str]] = None,
) -> Tuple[dict, str]:
    """
    Build a LangGraph / LangChain RunnableConfig dict that:
    - Names the run in LangSmith
    - Injects a Langfuse callback (if configured)
    - Carries a consistent trace_id for post-run scoring

    Returns (config_dict, trace_id).
    Pass config_dict as  `config=`  to graph.astream() / graph.invoke().
    """
    trace_id = str(uuid.uuid4())
    tags: List[str] = ["neuroresearch"]
    if deep_research:
        tags.append("deep_research")
    if extra_tags:
        tags.extend(extra_tags)

    callbacks: list = []
    lf_cb = _make_langfuse_callback(
        trace_id=trace_id,
        session_id=session_id,
        user_id=str(user_id),
        topic=topic,
    )
    if lf_cb:
        callbacks.append(lf_cb)

    config: dict = {
        "run_name": f"research:{topic[:60]}",
        "tags": tags,
        "metadata": {
            "topic":        topic,
            "user_id":      user_id,
            "session_id":   session_id,
            "deep_research": deep_research,
            "trace_id":     trace_id,
        },
    }
    if callbacks:
        config["callbacks"] = callbacks

    return config, trace_id


# ── Startup helper ────────────────────────────────────────────────────────────

def init_observability() -> None:
    """Activate all configured tracers.  Call once at app startup."""
    configure_langsmith()
    get_langfuse()          # initialise singleton and log status
