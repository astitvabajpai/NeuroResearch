import re
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a Quality Critic. Evaluate this research draft.

Topic: {topic}
Draft:
{draft}

Rate on Accuracy, Completeness, and Clarity (0.0 to 1.0).
Respond EXACTLY in this format with nothing else:
SCORE: <float>
FEEDBACK: <specific improvements needed>
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a Senior Academic Reviewer. Rigorously evaluate this research report.

Topic: {topic}
Draft:
{draft}

Evaluate on:
- Depth and comprehensiveness (are all major aspects covered?)
- Accuracy and use of evidence (are claims supported by specific data?)
- Structure and clarity (is it well-organized and readable?)
- Insight quality (does it go beyond surface-level observations?)
- Minimum length met (should be 800+ words for deep research)

Respond EXACTLY in this format with nothing else:
SCORE: <float between 0.0 and 1.0>
FEEDBACK: <detailed list of specific improvements needed>
""")


class CritiqueAgent(BaseAgent):
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
        model_id = state.get("critique_model") or self.model_id
        deep = state.get("deep_research", False)
        self._initialize(model_id, deep=deep)

        prompt = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain = prompt | self.llm
        response = chain.invoke({"topic": state["topic"], "draft": state["draft"]})
        text = response.content if hasattr(response, "content") else str(response)

        score_match    = re.search(r"SCORE:\s*(0\.\d+|1\.0|1|0)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.*)", text, re.DOTALL)

        score    = float(score_match.group(1)) if score_match else 0.5
        feedback = feedback_match.group(1).strip() if feedback_match else "Improve depth and detail."

        return {"critique_score": score, "critique_feedback": feedback}
