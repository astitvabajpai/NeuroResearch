from typing import TypedDict, Annotated, List, Optional
import operator


class ResearchState(TypedDict):
    topic: str
    research_notes: Annotated[List[str], operator.add]
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
    # MCP tool names available in this run (informational)
    mcp_tools: List[str]
    # Observability — trace IDs propagated for Langfuse scoring
    trace_id: Optional[str]
