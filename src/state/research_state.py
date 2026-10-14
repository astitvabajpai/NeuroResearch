from typing import TypedDict, Annotated, List, Optional, Dict
import operator


class ResearchState(TypedDict):
    topic:             str
    research_notes:    Annotated[List[str], operator.add]
    draft:             str
    critique_feedback: str
    critique_score:    float
    critique_rubric:   Dict[str, float]   # factual_support, coverage, structure, clarity
    iteration:         int
    max_iterations:    int
    # Per-agent model overrides
    research_model:    str
    writer_model:      str
    critique_model:    str
    # Mode flags
    deep_research:     bool
    # RAG — session-scoped vector store key + whether RAG index was built
    session_id:        str
    rag_enabled:       bool
    # MCP — tool names active in this run (informational + used by ResearchAgent)
    mcp_tools:         List[str]
    # Observability
    trace_id:          Optional[str]
