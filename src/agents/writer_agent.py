from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a Technical Writer. Write a well-structured research report.

Topic: {topic}
Research Notes:
{notes}

Write a report with these exact sections using ## markdown headings:
## Introduction
(2-3 paragraphs introducing the topic and its significance)

## Key Findings
(numbered list of 5+ specific findings with brief explanations)

## Analysis
(2-3 paragraphs analyzing patterns, implications, and connections between findings)

## Conclusion
(1-2 paragraphs summarizing insights and future outlook)

Be specific, factual, and cite details from the research notes.
Minimum 400 words.
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Senior Technical Writer producing a comprehensive research report.

Topic: {topic}
Research Notes:
{notes}

Write a detailed, publication-quality report with these exact sections using ## markdown headings:

## Executive Summary
(3-4 sentences summarizing the most important findings)

## Introduction
(3-4 paragraphs: background, significance, scope of this report)

## Current State of Research
(detailed overview of where the field stands today, 3-4 paragraphs)

## Key Findings
(numbered list of 8-10 findings, each with 2-3 sentences of explanation and evidence)

## Technical Deep Dive
(2-3 paragraphs analyzing the most complex or important aspects in detail)

## Applications & Real-World Impact
(2-3 paragraphs on practical uses, case studies, industry adoption)

## Challenges & Limitations
(numbered list of 4-5 current challenges with explanations)

## Future Directions
(2-3 paragraphs on emerging trends, open research questions, what comes next)

## Conclusion
(2-3 paragraphs synthesizing all findings and their broader significance)

Be highly specific, use data and statistics from the notes, minimum 800 words.
""")


class WriterAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
        if self.llm is None or target != self.model_id:
            self.model_id = target
            from src.tools.hg_llm import get_hf_llm
            self.llm = get_hf_llm(self.model_id, deep=deep)

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("writer_model") or self.model_id
        deep = state.get("deep_research", False)
        self._initialize(model_id, deep=deep)

        notes = "\n\n".join(state["research_notes"])
        prompt = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain = prompt | self.llm
        draft = chain.invoke({"topic": state["topic"], "notes": notes})
        draft_str = draft.content if hasattr(draft, "content") else str(draft)
        return {"draft": draft_str}
