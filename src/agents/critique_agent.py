"""
Critique Agent — multi-rubric quality scoring with source grounding check.

Prompt design principles:
- Critic role is adversarial by design: "find problems, not compliments"
- Each rubric dimension has calibration anchors with concrete failure examples,
  not just abstract descriptions
- Feedback must be prescriptive (what to change + how) not descriptive (what's wrong)
- The critic is shown the retrieved sources to enable genuine grounding checks
- Format enforcement: 5-line output is modelled explicitly to prevent free-form responses
- Score inflation guard: anchors are biased toward honest scoring, not encouragement
"""
import re
import logging
from typing import Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

logger = logging.getLogger(__name__)


STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are a rigorous peer reviewer. Your job is to identify weaknesses in a research report \
and score it honestly so the writer knows exactly what to fix. Do not be encouraging. \
Be accurate and specific.

TOPIC: {topic}

RETRIEVED SOURCES (the only factual material the writer had access to):
{sources}

REPORT TO EVALUATE:
{draft}

---

Score the report on four dimensions. Use the calibration anchors below — do not round up.

FACTUAL_SUPPORT
Measures: Are claims in the report backed by the retrieved sources above? \
Are inline citations present and do they match actual retrieved URLs?
  1.0 — Every factual claim in Key Findings cites a URL from the retrieved sources above
  0.8 — Most claims cited; 1–2 Key Findings lack citations
  0.6 — Several claims uncited or citations present but URLs not in retrieved sources
  0.4 — Fewer than half the Key Findings are properly cited
  0.2 — Almost no citations, or citations are clearly fabricated
  0.0 — No citations whatsoever

COVERAGE
Measures: Are all 4 required sections present (Introduction, Key Findings, Analysis, \
Conclusion)? Is the report at least 500 words? Does it cover the topic thoroughly?
  1.0 — All 4 sections complete, 500+ words, topic covered comprehensively
  0.8 — All sections present, slightly thin in one area
  0.6 — One section is weak or the report is 350–499 words
  0.4 — One section is missing or the report is under 350 words
  0.2 — Two or more sections missing
  0.0 — Report is a fragment

STRUCTURE
Measures: Logical flow, section headings correct, paragraphs coherent, \
no abrupt jumps or repetition.
  1.0 — Smooth flow, every section leads naturally to the next, no repetition
  0.8 — Good flow with one minor structural issue
  0.6 — One section feels out of place or has noticeable repetition
  0.4 — Multiple structural problems; hard to follow
  0.2 — Incoherent organisation
  0.0 — Unreadable

CLARITY
Measures: Are claims specific and concrete (names, numbers, dates, examples), \
or vague and generic?
  1.0 — Every paragraph contains specific, named facts or figures
  0.8 — Mostly specific; one or two vague passages
  0.6 — Mix of specific and generic; reader often left with "so what?"
  0.4 — Mostly generic; few concrete specifics
  0.2 — Almost entirely vague assertions
  0.0 — Pure filler, no informational content

---

FEEDBACK must:
- Be 1–3 sentences
- Name the specific sections or sentences that need fixing
- Say exactly what to add, change, or remove — not just "improve X"
- Start with the most important fix first

Bad feedback example: "The report needs more citations and better structure."
Good feedback example: "Key Findings 3 and 5 have no citations — add URLs from the \
retrieved sources. The Analysis section repeats the findings rather than synthesising \
them — replace Para 1 with a discussion of what the findings collectively imply."

---

Respond with EXACTLY these 5 lines and nothing else:
FACTUAL_SUPPORT: 0.85
COVERAGE: 0.90
STRUCTURE: 0.80
CLARITY: 0.75
FEEDBACK: [your specific, actionable feedback here]
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a rigorous peer reviewer evaluating a comprehensive deep-dive research report. \
Your job is to identify weaknesses honestly and precisely so the writer can produce a \
publication-quality revision. Do not be encouraging. Be accurate and specific.

TOPIC: {topic}

RETRIEVED SOURCES (the only factual material the writer had access to):
{sources}

REPORT TO EVALUATE:
{draft}

---

Score the report on four dimensions using the calibration anchors below. Do not round up.

FACTUAL_SUPPORT
Measures: Are claims backed by retrieved sources? Are inline citations present and accurate?
  1.0 — Every Key Finding and every factual claim in Current State cites a retrieved URL
  0.8 — Most claims cited; 1–2 Key Findings or Current State paras lack citations
  0.6 — Several Key Findings uncited, or some URLs are not in the retrieved sources
  0.4 — Fewer than half the Key Findings are cited
  0.2 — Almost no citations, or citations are fabricated
  0.0 — No citations at all

COVERAGE
Measures: All 9 sections present? (Executive Summary, Introduction, Current State, \
Key Findings, Technical Deep Dive, Applications, Challenges, Future Directions, Conclusion) \
900+ words? Topic covered with sufficient depth?
  1.0 — All 9 sections, 900+ words, comprehensive coverage
  0.8 — All 9 sections present, one is thin, 800+ words
  0.6 — One section missing or report is 650–799 words
  0.4 — Two sections missing or report is under 650 words
  0.2 — Three or more sections missing
  0.0 — Grossly incomplete

STRUCTURE
Measures: Logical progression across all 9 sections, headings correct, no repetition \
between sections (especially between Key Findings and Technical Deep Dive).
  1.0 — Perfect flow, each section builds on the last, no content repeated across sections
  0.8 — Good flow, minor overlap between one pair of sections
  0.6 — One section feels misplaced or there is noticeable repetition
  0.4 — Multiple structural problems
  0.2 — Incoherent
  0.0 — Unreadable

CLARITY
Measures: Concreteness and specificity throughout — names, dates, metrics, paper titles, \
company names, benchmarks.
  1.0 — Every paragraph in every section contains specific, named evidence
  0.8 — Mostly specific; one or two sections lapse into generality
  0.6 — Mix of specific and generic; Technical Deep Dive or Applications is vague
  0.4 — Mostly generic; feels like a summary, not an analysis
  0.2 — Almost no concrete specifics
  0.0 — No informational content

---

FEEDBACK must:
- Be 2–4 sentences
- Name the specific section and paragraph that needs fixing
- Say exactly what to add, change, or remove
- Prioritise: most impactful fix first

Bad example: "The report needs more depth and citations."
Good example: "Technical Deep Dive Para 2 describes tradeoffs in general terms — \
replace it with a specific comparison of two named approaches from the research notes. \
Key Findings 4, 7, and 9 have no citations — add URLs from the retrieved sources. \
Future Directions repeats content from Current State — rewrite it to discuss what \
is not yet possible but is actively being researched."

---

Respond with EXACTLY these 5 lines and nothing else:
FACTUAL_SUPPORT: 0.85
COVERAGE: 0.90
STRUCTURE: 0.80
CLARITY: 0.75
FEEDBACK: [your specific, actionable feedback here]
""")


def _parse_rubric_response(text: str) -> Tuple[float, Dict[str, float], str]:
    """
    Parse the 4-dimension rubric response.
    Returns (overall_score, rubric_dict, feedback_str).
    """
    logger.debug("[CritiqueAgent] Raw response: %s", text[:300])

    rubric: Dict[str, float] = {}
    dims = ["FACTUAL_SUPPORT", "COVERAGE", "STRUCTURE", "CLARITY"]

    for dim in dims:
        m = re.search(rf"{dim}\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val = val / 100.0
            rubric[dim.lower()] = max(0.0, min(1.0, val))
        else:
            rubric[dim.lower()] = 0.5

    # factual_support weighted highest — it's the RAG quality signal
    weights = {"factual_support": 0.35, "coverage": 0.30, "structure": 0.20, "clarity": 0.15}
    overall = sum(rubric.get(k, 0.5) * w for k, w in weights.items())
    overall = max(0.0, min(1.0, overall))

    m = re.search(r"FEEDBACK\s*[:=]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    feedback = m.group(1).strip() if m else text.strip()
    feedback = re.split(r"\n(?:FACTUAL|COVERAGE|STRUCTURE|CLARITY|SCORE)", feedback)[0].strip()
    if not feedback:
        feedback = "Add specific inline citations to Key Findings, expand thin sections, and ensure all claims are concrete."

    logger.info(
        "[CritiqueAgent] factual=%.2f coverage=%.2f structure=%.2f clarity=%.2f overall=%.2f",
        rubric.get("factual_support", 0),
        rubric.get("coverage", 0),
        rubric.get("structure", 0),
        rubric.get("clarity", 0),
        overall,
    )
    return overall, rubric, feedback


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

        sources = _extract_sources(state.get("research_notes", []))

        prompt   = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain    = prompt | self.llm
        response = chain.invoke({
            "topic":   state["topic"],
            "draft":   state["draft"],
            "sources": sources,
        })
        text = response.content if hasattr(response, "content") else str(response)

        overall, rubric, feedback = _parse_rubric_response(text)

        return {
            "critique_score":    overall,
            "critique_rubric":   rubric,
            "critique_feedback": feedback,
        }


def _extract_sources(notes: list) -> str:
    """Extract source URLs from research notes for the Critique Agent's grounding check."""
    if not notes:
        return "No sources retrieved."
    url_pattern = re.compile(r"https?://\S+")
    seen = set()
    urls = []
    for note in notes:
        for url in url_pattern.findall(note):
            url = url.rstrip(".,)")
            if url not in seen:
                seen.add(url)
                urls.append(url)
    if not urls:
        return "No source URLs found in research notes."
    return "\n".join(f"- {u}" for u in urls[:20])
