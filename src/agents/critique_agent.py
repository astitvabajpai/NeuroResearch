import re
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a calibrated research quality evaluator. Score this report fairly and accurately.

Topic: {topic}

Report to evaluate:
{draft}

Scoring rubric — evaluate each dimension and combine into a final score:

1. ACCURACY (0-1): Are the facts, statistics, and claims correct and specific?
   - 0.9-1.0: All claims are verifiable and precise with specific details
   - 0.7-0.9: Most claims are accurate with some vague statements
   - 0.5-0.7: Some inaccuracies or too many unsupported generalizations
   - Below 0.5: Significant factual issues or mostly vague

2. COMPLETENESS (0-1): Does it fully cover the topic?
   - 0.9-1.0: Comprehensive coverage of all key aspects with 500+ words
   - 0.7-0.9: Good coverage with minor gaps
   - 0.5-0.7: Missing important aspects or too brief
   - Below 0.5: Major gaps or very superficial

3. CLARITY (0-1): Is it well-written and well-structured?
   - 0.9-1.0: Excellent prose, clear headings, flows logically
   - 0.7-0.9: Generally clear with minor issues
   - 0.5-0.7: Some confusing parts or poor structure
   - Below 0.5: Difficult to follow

IMPORTANT CALIBRATION NOTES:
- A solid, specific, well-structured 500+ word report on a technical topic should score 0.80-0.90
- Only give below 0.7 if there are real problems (vague, too short, inaccurate)
- Do not penalize for not being a PhD dissertation — this is a research brief
- A report that covers all 4 sections with specific details deserves at least 0.82

Final score = average of the three dimensions.

Respond EXACTLY in this format — nothing else:
SCORE: <single float between 0.0 and 1.0>
FEEDBACK: <2-3 specific, actionable improvements for the next iteration>
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a calibrated senior research reviewer. Evaluate this comprehensive report fairly.

Topic: {topic}

Report to evaluate:
{draft}

Scoring rubric:

1. DEPTH & COMPREHENSIVENESS (0-1):
   - 0.9-1.0: All 9 sections present, each with substantive content, 900+ words
   - 0.7-0.9: Most sections present and substantive, minor gaps
   - 0.5-0.7: Some sections thin or missing
   - Below 0.5: Incomplete structure or very shallow

2. EVIDENCE QUALITY (0-1):
   - 0.9-1.0: Specific statistics, dates, named systems/papers/companies throughout
   - 0.7-0.9: Good use of evidence with some vague areas
   - 0.5-0.7: Limited evidence, mostly general statements
   - Below 0.5: No concrete evidence

3. INSIGHT QUALITY (0-1):
   - 0.9-1.0: Analysis goes beyond summarizing — shows connections, implications, nuance
   - 0.7-0.9: Some original analysis with good synthesis
   - 0.5-0.7: Mostly descriptive without analysis
   - Below 0.5: Surface-level only

4. CLARITY & STRUCTURE (0-1):
   - 0.9-1.0: Publication-quality prose, excellent organization
   - 0.7-0.9: Clear and well-organized
   - 0.5-0.7: Some structural or clarity issues

CALIBRATION:
- A thorough 900+ word report with specific evidence across all sections should score 0.82-0.92
- Reserve 0.95+ for truly exceptional reports with unique insight and comprehensive evidence
- Only give below 0.7 for reports with real structural or accuracy problems

Final score = average of the four dimensions.

Respond EXACTLY in this format — nothing else:
SCORE: <single float between 0.0 and 1.0>
FEEDBACK: <3-4 specific, actionable improvements addressing the weakest dimensions>
""")


class CritiqueAgent(BaseAgent):
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
        model_id = state.get("critique_model") or self.model_id
        deep     = state.get("deep_research", False)
        self._initialize(model_id, deep=deep)

        prompt   = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain    = prompt | self.llm
        response = chain.invoke({"topic": state["topic"], "draft": state["draft"]})
        text     = response.content if hasattr(response, "content") else str(response)

        score_match    = re.search(r"SCORE:\s*(0\.\d+|1\.0|1|0)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.*)", text, re.DOTALL)

        score    = float(score_match.group(1)) if score_match else 0.5
        feedback = feedback_match.group(1).strip() if feedback_match else "Improve depth and specificity."

        return {"critique_score": score, "critique_feedback": feedback}
