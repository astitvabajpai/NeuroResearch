from src.graph.research_graph import get_research_app

def run_research(topic: str):
    research_app = get_research_app()
    result = research_app.invoke({
        "topic": topic,
        "research_notes": [],
        "draft": "",
        "critique_feedback": "",
        "critique_score": 0.0,
        "iteration": 0,
        "max_iterations": 3,
        "final_report": ""
    })
    
    print(f"\n{'='*50}")
    print(f"FINAL REPORT (Score: {result['critique_score']:.2f})")
    print(f"Iterations: {result['iteration']}")
    print(f"{'='*50}\n")
    draft = result["draft"]
    # Handle both plain strings and LangChain message objects
    if hasattr(draft, "content"):
        draft = draft.content
    print(draft)
    return result

if __name__ == "__main__":
    run_research("Impact of transformers on protein folding prediction")