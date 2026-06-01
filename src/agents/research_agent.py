from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a Research Analyst. Extract factual, specific findings from the search results below.

Topic: {topic}
Search Results:
{search_results}

Extract exactly 5 key findings. Each must be a specific fact or insight directly relevant to the topic.
Output ONLY bullet points starting with "• ". No intro, no conclusion.
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Senior Research Analyst conducting an in-depth investigation.

Topic: {topic}
Search Results:
{search_results}

Perform a thorough analysis and extract 8-10 detailed findings covering:
1. Core concepts and definitions
2. Recent developments and breakthroughs
3. Key statistics, data points, and evidence
4. Different perspectives and debates in the field
5. Practical applications and real-world impact
6. Limitations, challenges, and open questions

For each finding write 2-3 sentences with specific details.
Output ONLY the numbered findings. No intro, no conclusion.
""")


class ResearchAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm = None
        self.search_tool = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
        if self.llm is None or target != self.model_id:
            self.model_id = target
            from src.tools.hg_llm import get_hf_llm
            self.llm = get_hf_llm(self.model_id, deep=deep)
        if self.search_tool is None:
            from src.tools.search_tool import get_search_tool
            self.search_tool = get_search_tool()

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("research_model") or self.model_id
        deep = state.get("deep_research", False)
        self._initialize(model_id, deep=deep)

        # Deep mode: run 2 searches (topic + "latest research on topic")
        if deep:
            r1 = self.search_tool.run(state["topic"])
            r2 = self.search_tool.run(f"latest research advances {state['topic']}")
            search_results = f"=== Search 1 ===\n{r1}\n\n=== Search 2 ===\n{r2}"
        else:
            search_results = self.search_tool.run(state["topic"])

        prompt = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain = prompt | self.llm
        notes_text = chain.invoke({
            "topic": state["topic"],
            "search_results": search_results,
        })
        notes_str = notes_text.content if hasattr(notes_text, "content") else str(notes_text)
        new_notes = state.get("research_notes", []) + [notes_str]
        return {"research_notes": new_notes, "iteration": state["iteration"] + 1}
