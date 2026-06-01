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


_research_app = None


def get_research_app():
    global _research_app
    if _research_app is None:
        _research_app = build_graph()
    return _research_app
