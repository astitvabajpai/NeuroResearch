"""
CLI / programmatic entry point.
Run directly:  python -m src.main
Or:            python src/main.py
"""

from src.graph.research_graph import build_initial_state, get_research_app
from src.tools.llm import DEFAULT_MODEL, DEFAULT_WRITER_MODEL, DEFAULT_CRITIC_MODEL


def run_research(
    topic: str,
    max_iterations: int = 3,
    research_model: str = None,
    writer_model: str = None,
    critique_model: str = None,
    deep_research: bool = False,
) -> dict:
    research_app  = get_research_app()
    initial_state = build_initial_state(
        topic=topic,
        max_iterations=max_iterations,
        research_model=research_model or DEFAULT_MODEL,
        writer_model=writer_model     or DEFAULT_WRITER_MODEL,
        critique_model=critique_model or DEFAULT_CRITIC_MODEL,
        deep_research=deep_research,
    )
    result = research_app.invoke(initial_state)

    draft = result.get("draft", "")
    if hasattr(draft, "content"):
        draft = draft.content

    print(f"\n{'='*50}")
    print(f"FINAL REPORT  (Score: {result.get('critique_score', 0):.2f}  |  "
          f"Iterations: {result.get('iteration', 0)})")
    print(f"{'='*50}\n")
    print(draft)
    return result


if __name__ == "__main__":
    run_research("Impact of transformers on protein folding prediction")
