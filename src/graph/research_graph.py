from langgraph.graph import StateGraph, END
from src.state.research_state import ResearchState
from src.agents.research_agent import ResearchAgent
from src.agents.writer_agent import WriterAgent
from src.agents.critique_agent import CritiqueAgent
from src.config.settings import get_settings

settings = get_settings()

from langgraph.graph import StateGraph, END
from src.state.research_state import ResearchState
from src.agents.research_agent import ResearchAgent
from src.agents.writer_agent import WriterAgent
from src.agents.critique_agent import CritiqueAgent
from src.config.settings import get_settings

settings = get_settings()

def build_graph():
    # Initialize agents lazily
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    critique_agent = CritiqueAgent()
    
    workflow = StateGraph(ResearchState)
    
    # Nodes with agent methods
    workflow.add_node("research", research_agent.invoke)
    workflow.add_node("write", writer_agent.invoke)
    workflow.add_node("critique", critique_agent.invoke)
    
    # Edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "critique")
    
    # Conditional self-correction loop
    def should_continue(state: ResearchState):
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if state["critique_score"] >= settings.QUALITY_THRESHOLD:
            return "finalize"
        return "revise"
    
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revise": "research",   # ← THE SELF-CORRECTION LOOP
            "finalize": END
        }
    )
    
    return workflow.compile()

# Create app lazily to avoid initialization issues
_research_app = None

def get_research_app():
    global _research_app
    if _research_app is None:
        try:
            _research_app = build_graph()
        except Exception as e:
            print(f"Error building graph: {e}")
            raise
    return _research_app

# For backward compatibility
research_app = None

def __getattr__(name):
    if name == "research_app":
        return get_research_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")