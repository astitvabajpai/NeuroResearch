from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are an expert Research Analyst. Your job is to extract rich, detailed, factual findings \
from the search results below and turn them into high-quality research notes.

Topic: {topic}

Search Results:
{search_results}

Instructions:
- Extract 5 key findings that are specific, factual, and directly relevant to the topic.
- Each finding must be 2-3 sentences long with concrete details, numbers, or examples.
- Include source URLs when present in the search results.
- Cover: core facts, recent developments, real-world applications, and key statistics.
- Do NOT write generic statements. Every finding must add unique, verifiable value.

Format — output ONLY this, nothing else:
• [Finding 1: specific fact with detail and evidence]
• [Finding 2: specific fact with detail and evidence]
• [Finding 3: specific fact with detail and evidence]
• [Finding 4: specific fact with detail and evidence]
• [Finding 5: specific fact with detail and evidence]
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Senior Research Analyst conducting a comprehensive investigation. \
Extract deep, publication-quality research notes from the search results below.

Topic: {topic}

Search Results:
{search_results}

Instructions:
- Extract 8-10 detailed findings covering ALL of the following dimensions:
  1. Core definitions and foundational concepts
  2. Historical context and how the field evolved
  3. Current state-of-the-art and recent breakthroughs (with dates/versions where available)
  4. Key statistics, benchmarks, and quantitative data
  5. Leading researchers, institutions, or companies driving progress
  6. Real-world applications and industry adoption (with specific examples)
  7. Open challenges, limitations, and unsolved problems
  8. Competing approaches or schools of thought
  9. Future directions and emerging trends
- Each finding must be 3-4 sentences with specific names, numbers, and citations.
- Include source URLs from the search results wherever available.
- Do NOT be vague. Precision and specificity are critical.

Format — output ONLY numbered findings, nothing else:
1. [Detailed finding with evidence, numbers, and sources]
2. [Detailed finding with evidence, numbers, and sources]
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
            r2 = self.search_tool.run(f"latest research advances {topic} 2024 2025")
            r3 = self.search_tool.run(f"{topic} applications examples case studies")
            search_results = (
                f"=== Web Search 1: Core Topic ===\n{r1}\n\n"
                f"=== Web Search 2: Latest Advances ===\n{r2}\n\n"
                f"=== Web Search 3: Applications ===\n{r3}"
            )
        else:
            r1 = self.search_tool.run(topic)
            r2 = self.search_tool.run(f"{topic} overview key facts")
            search_results = f"=== Search 1 ===\n{r1}\n\n=== Search 2 ===\n{r2}"

        prompt     = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain      = prompt | self.llm
        notes_text = chain.invoke({"topic": topic, "search_results": search_results})
        notes_str  = notes_text.content if hasattr(notes_text, "content") else str(notes_text)

        return {"research_notes": [notes_str], "iteration": state["iteration"] + 1}
