from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState


class ResearchAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm = None
        self.search_tool = None
        self.prompt = ChatPromptTemplate.from_template("""\
You are a Research Analyst. Analyze search results and extract key findings.

Topic: {topic}
Search Results:
{search_results}

Extract 3-5 key research findings or insights from the search results.
Format as bullet points.
""")

    def _initialize(self):
        if self.llm is None:
            from src.tools.hg_llm import get_hf_llm
            self.llm = get_hf_llm(self.model_id)
        if self.search_tool is None:
            from src.tools.search_tool import get_search_tool
            self.search_tool = get_search_tool()

    def invoke(self, state: ResearchState) -> dict:
        # Allow per-run model override from state
        model_id = state.get("research_model") or self.model_id
        if model_id != self.model_id or self.llm is None:
            self.model_id = model_id
            self.llm = None
        self._initialize()

        search_results = self.search_tool.run(state["topic"])
        chain = self.prompt | self.llm
        notes_text = chain.invoke({
            "topic": state["topic"],
            "search_results": search_results,
        })
        notes_str = notes_text.content if hasattr(notes_text, "content") else str(notes_text)
        new_notes = state.get("research_notes", []) + [notes_str]
        return {"research_notes": new_notes, "iteration": state["iteration"] + 1}
