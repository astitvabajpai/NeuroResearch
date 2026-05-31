import re
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState


class CritiqueAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm = None
        self.prompt = ChatPromptTemplate.from_template("""\
You are a Quality Critic. Evaluate the research draft strictly.

Topic: {topic}
Draft:
{draft}

Rate on Accuracy, Completeness, and Clarity.
Respond EXACTLY in this format:
SCORE: <float between 0.0 and 1.0>
FEEDBACK: <specific improvements needed>
""")

    def _initialize(self):
        if self.llm is None:
            from src.tools.hg_llm import get_hf_llm
            self.llm = get_hf_llm(self.model_id)

    def invoke(self, state: ResearchState) -> dict:
        model_id = state.get("critique_model") or self.model_id
        if model_id != self.model_id or self.llm is None:
            self.model_id = model_id
            self.llm = None
        self._initialize()

        chain = self.prompt | self.llm
        response = chain.invoke({"topic": state["topic"], "draft": state["draft"]})
        text = response.content if hasattr(response, "content") else str(response)

        score_match    = re.search(r"SCORE:\s*(0\.\d+|1\.0|1|0)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.*)", text, re.DOTALL)

        score    = float(score_match.group(1)) if score_match else 0.5
        feedback = feedback_match.group(1).strip() if feedback_match else "Improve clarity and depth."

        return {"critique_score": score, "critique_feedback": feedback}
