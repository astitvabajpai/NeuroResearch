from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are an expert Technical Writer. Write a high-quality, well-structured research report \
that would score 0.95+ on accuracy, completeness, and clarity.

Topic: {topic}

Research Notes (use ALL of these findings — do not ignore any):
{notes}

Critique from previous iteration (if any — address every point):
{feedback}

Requirements — your report MUST:
✓ Use ALL findings from the research notes with specific details
✓ Include concrete numbers, statistics, dates, and named examples
✓ Be written in clear, professional prose — no vague generalities
✓ Reach at least 500 words
✓ Use EXACTLY these section headings (## markdown):

## Introduction
Write 3 paragraphs covering: what the topic is, why it matters today, \
and the scope of this report. Name specific applications or use cases.

## Key Findings
Write a numbered list of 5-7 findings. Each item must be 2-3 sentences \
with specific facts, statistics, or named examples from the research notes. \
Do not write one-liners.

## Analysis
Write 3 paragraphs that: (1) identify patterns across the findings, \
(2) explain implications for practitioners or researchers, \
(3) connect findings to broader trends in the field.

## Conclusion
Write 2 paragraphs: (1) synthesize the most important insights, \
(2) give a concrete outlook — what will happen next in this field.

Start writing the report directly. Do not include any preamble.
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Senior Technical Writer producing a publication-quality comprehensive research report \
that would score 0.95+ on depth, accuracy, evidence quality, and clarity.

Topic: {topic}

Research Notes (synthesize ALL of these — do not omit any findings):
{notes}

Critique from previous iteration (if any — address every single point):
{feedback}

Requirements — your report MUST:
✓ Synthesize ALL research findings with specific details, numbers, and named sources
✓ Include statistics, benchmark results, dates, version numbers, and named examples
✓ Write in authoritative, publication-quality prose
✓ Reach at least 900 words
✓ Use EXACTLY these section headings (## markdown):

## Executive Summary
3-4 sentences. State the most important conclusion and 2-3 key numbers or facts.

## Introduction
4 paragraphs: (1) background and context, (2) why this matters now, \
(3) scope and methodology of this report, (4) what the reader will learn.

## Current State of the Field
4 paragraphs on where things stand today. Name specific systems, papers, \
companies, or technologies. Include dates and version numbers where relevant.

## Key Findings
Numbered list of 8-10 findings. Each must be 3-4 sentences with evidence. \
Do not write vague statements — every finding must be verifiable and specific.

## Technical Deep Dive
3 paragraphs analyzing the most technically complex or novel aspects. \
Explain mechanisms, architectures, or methods with precision.

## Applications & Real-World Impact
3 paragraphs with specific deployment examples, company names, \
case studies, and quantitative outcomes where available.

## Challenges & Limitations
Numbered list of 4-5 challenges with specific explanations of why they are hard \
and what is being done to address them.

## Future Directions
3 paragraphs on concrete next steps: what research is underway, \
what will likely happen in 1-3 years, and what open questions remain.

## Conclusion
3 paragraphs synthesizing everything: key takeaways, significance, \
and a clear-eyed assessment of where the field is headed.

Start writing the report directly. Do not include any preamble or meta-commentary.
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
        # Pass prior critique feedback so the writer can address it directly
        feedback = state.get("critique_feedback", "").strip()
        feedback = feedback if feedback else "This is the first draft — focus on completeness and detail."

        prompt    = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain     = prompt | self.llm
        draft     = chain.invoke({"topic": state["topic"], "notes": notes, "feedback": feedback})
        draft_str = draft.content if hasattr(draft, "content") else str(draft)
        return {"draft": draft_str}
