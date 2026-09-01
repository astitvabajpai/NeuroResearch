import re
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
Score this research report on a scale of 0.0 to 1.0.

Topic: {topic}

Report:
{draft}

Scoring guide:
- 0.90-1.00: Excellent — specific facts, all 4 sections present, 500+ words, clear analysis
- 0.80-0.89: Good — mostly complete with minor gaps in detail or depth  
- 0.70-0.79: Fair — some sections thin or lacking specific evidence
- Below 0.70: Needs significant improvement — major gaps or too short

Respond in this exact format with nothing else:
SCORE: [number between 0.0 and 1.0]
FEEDBACK: [2-3 specific improvements for the next draft]
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
Score this comprehensive research report on a scale of 0.0 to 1.0.

Topic: {topic}

Report:
{draft}

Scoring guide:
- 0.90-1.00: All 9 sections present, 900+ words, specific evidence throughout, strong analysis
- 0.80-0.89: Most sections complete with good evidence, minor gaps
- 0.70-0.79: Some sections thin or missing, limited specific evidence
- Below 0.70: Major sections missing or very superficial

Respond in this exact format with nothing else:
SCORE: [number between 0.0 and 1.0]
FEEDBACK: [3-4 specific improvements for the next draft]
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
