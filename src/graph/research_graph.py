"""
LangGraph research pipeline.

Graph topology:
    research → write → critique ──(score >= threshold or max iter)──► END
                  ▲                        │
                  └──────── revise ────────┘
"""

from langgraph.graph import StateGraph, END
from src.state.research_state import ResearchState
from src.agents.research_agent import ResearchAgent
from src.agents.writer_agent import WriterAgent
from src.agents.critique_agent import CritiqueAgent


def build_graph():
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


# Singleton — built once, reused across requests
_research_app = None


def get_research_app():
    global _research_app
    if _research_app is None:
        _research_app = build_graph()
    return _research_app


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
