"""
LangGraph research pipeline.

Graph topology:
    research → write → critique ──(score >= threshold or max iter)──► END
                  ▲                        │
                  └──────── revise ────────┘

NOTE: The graph is rebuilt on every call to get_research_app() so that
agent instances are always fresh and model rotation always works correctly.
"""

from langgraph.graph import StateGraph, END
from src.state.research_state import ResearchState


def build_graph():
    # Import agents fresh each build so they pick up latest code
    from src.agents.research_agent import ResearchAgent
    from src.agents.writer_agent   import WriterAgent
    from src.agents.critique_agent import CritiqueAgent

    research_agent = ResearchAgent()
    writer_agent   = WriterAgent()
    critique_agent = CritiqueAgent()

    workflow = StateGraph(ResearchState)
    workflow.add_node("research", research_agent.invoke)
    workflow.add_node("write",    writer_agent.invoke)
    workflow.add_node("critique", critique_agent.invoke)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write",    "critique")

    def should_continue(state: ResearchState) -> str:
        from src.config.settings import get_settings
        threshold = get_settings().QUALITY_THRESHOLD
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if state["critique_score"] >= threshold:
            return "finalize"
        return "revise"

    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {"revise": "research", "finalize": END},
    )

    return workflow.compile()


def get_research_app():
    """Always build fresh — no singleton — so agents are never stale."""
    return build_graph()


def _reset_app():
    """No-op kept for backward compatibility with debug scripts."""
    pass


def build_initial_state(
    topic: str,
    max_iterations: int,
    research_model: str,
    writer_model: str,
    critique_model: str,
    deep_research: bool = False,
    trace_id: str | None = None,
) -> dict:
    from src.tools.llm import DEFAULT_MODEL, DEFAULT_WRITER_MODEL, DEFAULT_CRITIC_MODEL
    return {
        "topic":             topic,
        "research_notes":    [],
        "draft":             "",
        "critique_feedback": "",
        "critique_score":    0.0,
        "iteration":         0,
        "max_iterations":    max_iterations,
        "final_report":      "",
        "research_model":    research_model or DEFAULT_MODEL,
        "writer_model":      writer_model   or DEFAULT_WRITER_MODEL,
        "critique_model":    critique_model or DEFAULT_CRITIC_MODEL,
        "deep_research":     deep_research,
        "mcp_tools":         [],
        "trace_id":          trace_id,
    }
