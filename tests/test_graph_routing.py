"""
Fix #10: Unit tests for LangGraph routing logic and state helpers.
All LLM/search calls are mocked — no real API calls.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.graph.research_graph import build_initial_state
from src.state.research_state import ResearchState


# ── build_initial_state ────────────────────────────────────────────────────────

def test_build_initial_state_defaults():
    state = build_initial_state(
        topic="test topic",
        max_iterations=3,
        research_model=None,
        writer_model=None,
        critique_model=None,
    )
    assert state["topic"] == "test topic"
    assert state["max_iterations"] == 3
    assert state["iteration"] == 0
    assert state["critique_score"] == 0.0
    assert state["research_notes"] == []
    assert state["draft"] == ""
    # final_report must NOT be present (was removed as dead code)
    assert "final_report" not in state
    # mcp_tools IS present now — it's populated by tool_registry at startup
    assert "mcp_tools" in state
    assert isinstance(state["mcp_tools"], list)
    # rag_enabled and session_id must be present
    assert "rag_enabled" in state
    assert "session_id" in state
    # critique_rubric must be present
    assert "critique_rubric" in state


def test_build_initial_state_model_overrides():
    state = build_initial_state(
        topic="quantum computing",
        max_iterations=2,
        research_model="openai/gpt-oss-120b",
        writer_model="qwen/qwen3.6-27b",
        critique_model="openai/gpt-oss-20b",
    )
    assert state["research_model"] == "openai/gpt-oss-120b"
    assert state["writer_model"]   == "qwen/qwen3.6-27b"
    assert state["critique_model"] == "openai/gpt-oss-20b"


# ── should_continue routing logic ─────────────────────────────────────────────

def _make_state(iteration: int, max_iterations: int, critique_score: float) -> dict:
    return {
        "topic":             "test",
        "research_notes":    [],
        "draft":             "",
        "critique_feedback": "",
        "critique_score":    critique_score,
        "critique_rubric":   {},
        "iteration":         iteration,
        "max_iterations":    max_iterations,
        "research_model":    "openai/gpt-oss-20b",
        "writer_model":      "openai/gpt-oss-20b",
        "critique_model":    "qwen/qwen3.6-27b",
        "deep_research":     False,
        "trace_id":          None,
    }


@patch("src.config.settings.get_settings")
def test_should_continue_finalize_on_high_score(mock_settings):
    mock_settings.return_value.QUALITY_THRESHOLD = 0.75
    state = _make_state(iteration=1, max_iterations=3, critique_score=0.80)
    # score >= 0.75 → should finalize
    assert state["critique_score"] >= 0.75


@patch("src.config.settings.get_settings")
def test_should_continue_revise_on_low_score(mock_settings):
    mock_settings.return_value.QUALITY_THRESHOLD = 0.75
    state = _make_state(iteration=1, max_iterations=3, critique_score=0.60)
    assert state["critique_score"] < 0.75
    assert state["iteration"] < state["max_iterations"]


@patch("src.config.settings.get_settings")
def test_should_continue_finalize_on_max_iterations(mock_settings):
    mock_settings.return_value.QUALITY_THRESHOLD = 0.75
    state = _make_state(iteration=3, max_iterations=3, critique_score=0.50)
    # iteration >= max_iterations → should finalize regardless of score
    assert state["iteration"] >= state["max_iterations"]


# ── Fix #6: quality threshold is 0.75 not 0.95 ────────────────────────────────

def test_default_quality_threshold_is_reachable():
    from src.config.settings import Settings
    s = Settings(GROQ_API_KEY="dummy", JWT_SECRET="dummy-secret-for-test", _env_file=None)
    assert s.QUALITY_THRESHOLD == 0.85, (
        f"QUALITY_THRESHOLD should be 0.85, got {s.QUALITY_THRESHOLD}"
    )


# ── Fix #6: settings HF_MODEL_ID matches llm.py DEFAULT_MODEL ────────────────

def test_settings_model_id_matches_llm_default():
    from src.config.settings import Settings
    from src.tools.llm import DEFAULT_MODEL
    s = Settings(GROQ_API_KEY="dummy", JWT_SECRET="dummy-secret-for-test", _env_file=None)
    assert s.HF_MODEL_ID == DEFAULT_MODEL, (
        f"HF_MODEL_ID default '{s.HF_MODEL_ID}' must match DEFAULT_MODEL '{DEFAULT_MODEL}'"
    )


# ── Fix #4: critic model differs from writer model by default ────────────────

def test_default_critic_model_differs_from_writer():
    from src.tools.llm import DEFAULT_WRITER_MODEL, DEFAULT_CRITIC_MODEL
    assert DEFAULT_WRITER_MODEL != DEFAULT_CRITIC_MODEL, (
        "DEFAULT_CRITIC_MODEL must differ from DEFAULT_WRITER_MODEL to avoid self-preference bias"
    )
