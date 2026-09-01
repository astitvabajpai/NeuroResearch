import re
import logging
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

logger = logging.getLogger(__name__)

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are evaluating a research report. Read the report below and give it a quality score.

Topic: {topic}

Report:
{draft}

Evaluate the report on accuracy, completeness, and clarity.

Your response MUST follow this exact format (two lines only):
SCORE: 0.85
FEEDBACK: The report needs more specific statistics and a stronger conclusion.

Replace 0.85 with your actual score between 0.0 and 1.0.
Replace the feedback text with 2-3 specific improvements.

Scoring reference:
- 0.90 to 1.00 = excellent, specific facts, all sections complete, 500+ words
- 0.80 to 0.89 = good, mostly complete with minor gaps
- 0.70 to 0.79 = fair, some sections thin or lacking evidence
- below 0.70 = needs major improvement

Respond with ONLY the two lines shown above. Nothing else.
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are evaluating a comprehensive research report. Read the report below and give it a quality score.

Topic: {topic}

Report:
{draft}

Evaluate the report on depth, evidence quality, insight, and clarity.

Your response MUST follow this exact format (two lines only):
SCORE: 0.85
FEEDBACK: The report needs more specific statistics and a stronger technical deep dive section.

Replace 0.85 with your actual score between 0.0 and 1.0.
Replace the feedback text with 3-4 specific improvements.

Scoring reference:
- 0.90 to 1.00 = all 9 sections present, 900+ words, specific evidence throughout
- 0.80 to 0.89 = most sections complete with good evidence, minor gaps
- 0.70 to 0.79 = some sections thin or missing, limited specific evidence
- below 0.70 = major sections missing or very superficial

Respond with ONLY the two lines shown above. Nothing else.
""")


def _parse_score_and_feedback(text: str) -> tuple[float, str]:
    """
    Robustly parse SCORE and FEEDBACK from LLM output.
    Handles many formats the model might use.
    """
    # Log the raw response for debugging
    logger.info("[CritiqueAgent] Raw response: %s", text[:200])

    # Try to find score — many possible formats:
    # SCORE: 0.85  |  Score: 0.85  |  **SCORE**: 0.85  |  score=0.85  |  0.85/1.0  |  85%
    score = None

    # Standard format first
    m = re.search(r"SCORE\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        score = float(m.group(1))

    # Percentage format (85% → 0.85)
    if score is None:
        m = re.search(r"([0-9]+)\s*%", text)
        if m:
            score = float(m.group(1)) / 100.0

    # x/1.0 or x/10 format
    if score is None:
        m = re.search(r"([0-9]*\.?[0-9]+)\s*/\s*(1\.?0?|10)", text)
        if m:
            val   = float(m.group(1))
            denom = float(m.group(2))
            score = val / denom if denom == 10 else val

    # Any float between 0 and 1 in the text
    if score is None:
        nums = re.findall(r"\b(0\.\d+|1\.0)\b", text)
        if nums:
            score = float(nums[0])

    # Clamp to valid range
    if score is not None:
        score = max(0.0, min(1.0, score))
    else:
        # Could not parse — default to 0.5 (neutral, not 0)
        logger.warning("[CritiqueAgent] Could not parse score from: %s", text[:100])
        score = 0.5

    # Parse feedback
    m = re.search(r"FEEDBACK\s*[:=]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    feedback = m.group(1).strip() if m else text.strip()

    # Trim feedback to remove any trailing SCORE lines if they appear after
    feedback = re.split(r"\nSCORE", feedback)[0].strip()

    if not feedback:
        feedback = "Improve depth and add more specific details."

    return score, feedback


class CritiqueAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm      = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
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

        score, feedback = _parse_score_and_feedback(text)
        logger.info("[CritiqueAgent] Parsed score=%.2f feedback=%s", score, feedback[:80])

        return {"critique_score": score, "critique_feedback": feedback}
