from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a Research Analyst. Read the search results below and extract exactly 5 key findings about the topic.

Topic: {topic}

Search Results:
{search_results}

Rules:
- Each finding must be 2-3 sentences with specific facts, numbers, or examples.
- Include source URLs from the search results where available.
- No intro, no conclusion, no meta-commentary.

Output exactly 5 bullet points:
• [finding 1]
• [finding 2]
• [finding 3]
• [finding 4]
• [finding 5]
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Research Analyst. Read the search results below and extract 8-10 detailed findings about the topic.

Topic: {topic}

Search Results:
{search_results}

Rules:
- Each finding must be 3-4 sentences with specific names, dates, numbers, and evidence.
- Include source URLs from the search results where available.
- Cover: definitions, recent developments, statistics, applications, challenges, future trends.
- No intro, no conclusion, no meta-commentary.

Output numbered findings only:
1. [finding]
2. [finding]
...
""")


class ResearchAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id    = model_id
        self.llm         = None
        self.search_tool = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
        if self.llm is None or target != self.model_id:
            self.model_id = target
            from src.tools.llm import get_llm
            self.llm = get_llm(self.model_id, deep=deep)
        if self.search_tool is None:
            from src.tools.search_tool import get_search_tool
            self.search_tool = get_search_tool()

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("research_model") or self.model_id
        deep     = state.get("deep_research", False)
        topic    = state["topic"]
        self._initialize(model_id, deep=deep)

        if deep:
            r1 = self.search_tool.run(topic)
            r2 = self.search_tool.run(f"latest {topic} 2024 2025")
            r3 = self.search_tool.run(f"{topic} applications examples")
            search_results = f"=== Search 1 ===\n{r1}\n\n=== Search 2 ===\n{r2}\n\n=== Search 3 ===\n{r3}"
        else:
            r1 = self.search_tool.run(topic)
            r2 = self.search_tool.run(f"{topic} key facts overview")
            search_results = f"=== Search 1 ===\n{r1}\n\n=== Search 2 ===\n{r2}"

        prompt     = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain      = prompt | self.llm
        notes_text = chain.invoke({"topic": topic, "search_results": search_results})
        notes_str  = notes_text.content if hasattr(notes_text, "content") else str(notes_text)

        return {"research_notes": [notes_str], "iteration": state["iteration"] + 1}
