from src.graph.research_graph import get_research_app
from src.tools.hg_llm import DEFAULT_MODEL


def run_research(topic: str, max_iterations: int = 3,
                 research_model: str = None,
                 writer_model: str = None,
                 critique_model: str = None):
    research_app = get_research_app()
    result = research_app.invoke({
        "topic":            topic,
        "research_notes":   [],
        "draft":            "",
        "critique_feedback":"",
        "critique_score":   0.0,
        "iteration":        0,
        "max_iterations":   max_iterations,
        "final_report":     "",
        "research_model":   research_model or DEFAULT_MODEL,
        "writer_model":     writer_model   or DEFAULT_MODEL,
        "critique_model":   critique_model or DEFAULT_MODEL,
    })

    print(f"\n{'='*50}")
    print(f"FINAL REPORT (Score: {result['critique_score']:.2f})")
    print(f"Iterations: {result['iteration']}")
    print(f"{'='*50}\n")
    draft = result["draft"]
    if hasattr(draft, "content"):
        draft = draft.content
    print(draft)
    return result


if __name__ == "__main__":
    run_research("Impact of transformers on protein folding prediction")
