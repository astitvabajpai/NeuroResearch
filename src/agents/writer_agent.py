from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState


class WriterAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm = None
        self.prompt = ChatPromptTemplate.from_template("""\
You are a Technical Writer. Write a clear, well-structured research report.

Topic: {topic}
Research Notes:
{notes}

Write a comprehensive draft with:
- A clear Title
- Introduction
- Key Findings (numbered list)
- Analysis
- Conclusion

Use markdown formatting with ## headings.
""")

    def _initialize(self):
        if self.llm is None:
            from src.tools.hg_llm import get_hf_llm
            self.llm = get_hf_llm(self.model_id)

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("writer_model") or self.model_id
        if model_id != self.model_id or self.llm is None:
            self.model_id = model_id
            self.llm = None
        self._initialize()

        notes = "\n".join(state["research_notes"])
        chain = self.prompt | self.llm
        draft = chain.invoke({"topic": state["topic"], "notes": notes})
        draft_str = draft.content if hasattr(draft, "content") else str(draft)
        return {"draft": draft_str}
