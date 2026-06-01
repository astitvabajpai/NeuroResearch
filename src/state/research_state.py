from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    topic: str
    research_notes: Annotated[list[str], operator.add]
    draft: str
    critique_feedback: str
    critique_score: float
    iteration: int
    max_iterations: int
    final_report: str
    # Per-agent model overrides
    research_model: str
    writer_model: str
    critique_model: str
    # Deep research mode
    deep_research: bool