import logging
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

logger = logging.getLogger(__name__)

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
Write a research report on the topic below using the provided research notes.

Topic: {topic}

Research Notes:
{notes}

Previous critique to address: {feedback}

Write the report using these exact section headings:

## Introduction
Three paragraphs. What the topic is, why it matters, and what this report covers.

## Key Findings
A numbered list of 5-7 findings. Each item is 2-3 sentences with specific facts and examples from the notes.

## Analysis
Three paragraphs analyzing patterns, implications, and connections between the findings.

## Conclusion
Two paragraphs summarizing key insights and the outlook for this topic.

Requirements: minimum 500 words, use specific details from the research notes, write in clear professional prose.
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
Write a comprehensive research report on the topic below using the provided research notes.

Topic: {topic}

Research Notes:
{notes}

Previous critique to address: {feedback}

Write the report using these exact section headings:

## Executive Summary
Three sentences summarizing the most important findings and their significance.

## Introduction
Four paragraphs covering background, significance, scope of this report, and what the reader will learn.

## Current State of the Field
Four paragraphs on where things stand today with specific systems, papers, companies, and dates.

## Key Findings
A numbered list of 8-10 findings. Each item is 3-4 sentences with specific evidence and examples.

## Technical Deep Dive
Three paragraphs analyzing the most complex aspects in detail.

## Applications & Real-World Impact
Three paragraphs with specific deployment examples, company names, and outcomes.

## Challenges & Limitations
A numbered list of 4-5 challenges with explanations of why they are hard.

## Future Directions
Three paragraphs on what is coming next with concrete examples and timelines.

## Conclusion
Three paragraphs synthesizing everything and giving a clear assessment of where the field is headed.

Requirements: minimum 900 words, use specific details and numbers from the research notes, write in clear professional prose.
""")


class WriterAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm      = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
        if self.llm is None or target != self.model_id:
            self.model_id = target
            from src.tools.llm import get_llm
            self.llm = get_llm(self.model_id, deep=deep)

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("writer_model") or self.model_id
        deep     = state.get("deep_research", False)
        self._initialize(model_id, deep=deep)

        notes    = "\n\n".join(state["research_notes"])
        feedback = state.get("critique_feedback", "").strip()
        if not feedback:
            feedback = "First draft — focus on completeness and use all research notes."

        prompt    = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain     = prompt | self.llm
        draft     = chain.invoke({"topic": state["topic"], "notes": notes, "feedback": feedback})
        draft_str = draft.content if hasattr(draft, "content") else str(draft)

        # Safety: if output is empty or just whitespace, return a note
        if not draft_str.strip():
            logger.warning("[WriterAgent] LLM returned empty draft")
            draft_str = f"## Introduction\n\nResearch report on: {state['topic']}\n\n(Draft generation failed — please retry)"

        return {"draft": draft_str}
