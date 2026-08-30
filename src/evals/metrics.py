"""
Eval metrics — deterministic + LLM-judge metrics.

Deterministic (no LLM, instant):
  - critique_score        : raw Critique Agent score
  - word_count            : draft length vs target
  - section_coverage      : required ## headings present
  - source_density        : % notes containing URLs
  - iteration_efficiency  : convergence speed
  - finding_count         : bullet/numbered items extracted

LLM-judge (requires HF_API_TOKEN, ~1 extra LLM call per eval):
  - faithfulness          : are report claims grounded in research notes?
  - retrieval_precision   : fraction of retrieved content matching gold facts
  - retrieval_recall      : fraction of gold facts found in retrieved content

All metrics return float in [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ResearchResult:
    topic: str
    draft: str
    research_notes: List[str]
    critique_score: float
    iterations: int
    deep_research: bool = False
    session_id: str = ""
    trace_id: str = ""
    # Gold data (from EvalFixture) — populated by harness before scoring
    gold_facts: List[str] = field(default_factory=list)
    gold_sources: List[str] = field(default_factory=list)


# ── Deterministic metrics ─────────────────────────────────────────────────────

def metric_critique_score(result: ResearchResult) -> float:
    return max(0.0, min(1.0, result.critique_score))


def metric_word_count(result: ResearchResult) -> float:
    target = 800 if result.deep_research else 400
    return min(1.0, len(result.draft.split()) / target)


def metric_section_coverage(result: ResearchResult) -> float:
    standard = {"introduction", "key findings", "analysis", "conclusion"}
    deep = {
        "executive summary", "introduction", "current state of research",
        "key findings", "technical deep dive", "applications",
        "challenges", "future directions", "conclusion",
    }
    expected = deep if result.deep_research else standard
    headings = {
        h.strip().lower()
        for h in re.findall(r"^##\s+(.+)$", result.draft, re.MULTILINE)
    }
    matched = sum(1 for exp in expected if any(exp in h for h in headings))
    return matched / len(expected) if expected else 1.0


def metric_source_density(result: ResearchResult) -> float:
    if not result.research_notes:
        return 0.0
    url_pat = re.compile(r"https?://\S+")
    sourced = sum(1 for n in result.research_notes if url_pat.search(n))
    return sourced / len(result.research_notes)


def metric_iteration_efficiency(result: ResearchResult) -> float:
    max_iter = 3
    if result.iterations <= 1:
        return 1.0
    if result.critique_score >= 0.8:
        return max(0.0, 1.0 - (result.iterations - 1) / max_iter)
    return 0.0


def metric_finding_count(result: ResearchResult) -> float:
    target = 8 if result.deep_research else 5
    count  = sum(
        len(re.findall(r"^[•\-\*]|\b\d+\.", note, re.MULTILINE))
        for note in result.research_notes
    )
    return min(1.0, count / target)


# ── Retrieval metrics (deterministic, uses gold_facts / gold_sources) ─────────

def metric_retrieval_precision(result: ResearchResult) -> float:
    """
    Fraction of gold_facts that appear anywhere in the research notes.
    Measures: did the search tool retrieve content containing expected facts?
    Returns 1.0 if no gold_facts defined (not applicable).
    """
    if not result.gold_facts:
        return 1.0
    combined_notes = "\n".join(result.research_notes).lower()
    hits = sum(1 for fact in result.gold_facts if fact.lower() in combined_notes)
    return hits / len(result.gold_facts)


def metric_retrieval_recall(result: ResearchResult) -> float:
    """
    Fraction of gold_sources (domain keywords) that appear in research notes.
    Measures: did the agent retrieve from expected authoritative sources?
    Returns 1.0 if no gold_sources defined.
    """
    if not result.gold_sources:
        return 1.0
    combined_notes = "\n".join(result.research_notes).lower()
    hits = sum(1 for src in result.gold_sources if src.lower() in combined_notes)
    return hits / len(result.gold_sources)


def metric_gold_fact_coverage(result: ResearchResult) -> float:
    """
    Fraction of gold_facts that appear in the final DRAFT (not just notes).
    Measures: did the Writer Agent actually include key facts in the report?
    """
    if not result.gold_facts:
        return 1.0
    draft_lower = result.draft.lower()
    hits = sum(1 for fact in result.gold_facts if fact.lower() in draft_lower)
    return hits / len(result.gold_facts)


# ── Faithfulness — LLM judge ──────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """\
You are an evaluator assessing whether a research report is faithful to its source material.

RESEARCH NOTES (retrieved source material):
{notes}

FINAL REPORT (generated from the notes):
{draft}

Task: Rate how well the final report is grounded in the research notes.
- Score 1.0 = every claim in the report is supported by the research notes
- Score 0.5 = roughly half the claims are supported; some hallucination present
- Score 0.0 = report contains mostly hallucinated content not found in notes

Respond ONLY with:
FAITHFULNESS_SCORE: <float between 0.0 and 1.0>
REASON: <one sentence explaining the score>
"""


def metric_faithfulness(result: ResearchResult, llm=None) -> float:
    """
    LLM-judge faithfulness: are the report's claims grounded in research notes?

    If `llm` is None, attempts to create one from settings.
    Returns 0.5 (neutral) on any error so it doesn't tank the overall score.
    """
    if not result.research_notes or not result.draft:
        return 0.5

    notes_text = "\n\n".join(result.research_notes)[:3000]  # trim for token limit
    draft_text = result.draft[:2000]

    prompt_text = FAITHFULNESS_PROMPT.format(
        notes=notes_text,
        draft=draft_text,
    )

    try:
        if llm is None:
            from src.tools.hg_llm import get_hf_llm
            llm = get_hf_llm(deep=False)

        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt_text)])
        text     = response.content if hasattr(response, "content") else str(response)

        match = re.search(r"FAITHFULNESS_SCORE:\s*(0\.\d+|1\.0|1|0)", text)
        return float(match.group(1)) if match else 0.5

    except Exception:
        return 0.5   # fail open — don't block the eval run


# ── Aggregate scorer ──────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    overall: float
    breakdown: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Overall: {self.overall:.3f}"]
        for k, v in self.breakdown.items():
            if v is not None:
                lines.append(f"  {k}: {v:.3f}")
            else:
                lines.append(f"  {k}: (not run)")
        return "\n".join(lines)


# Weights — deterministic metrics only (faithfulness scored separately)
_DETERMINISTIC_METRICS: List[tuple] = [
    ("critique_score",       metric_critique_score,      0.25),
    ("word_count",           metric_word_count,           0.10),
    ("section_coverage",     metric_section_coverage,     0.15),
    ("source_density",       metric_source_density,       0.05),
    ("iteration_efficiency", metric_iteration_efficiency, 0.10),
    ("finding_count",        metric_finding_count,        0.05),
    ("retrieval_precision",  metric_retrieval_precision,  0.10),
    ("retrieval_recall",     metric_retrieval_recall,     0.10),
    ("gold_fact_coverage",   metric_gold_fact_coverage,   0.10),
]
# faithfulness adds 0.0 weight here — computed separately and injected
_FAITHFULNESS_WEIGHT = 0.20


def score_result(
    result: ResearchResult,
    run_faithfulness: bool = False,
    llm=None,
) -> EvalScore:
    """
    Compute a weighted overall score.

    Args:
        result            : ResearchResult to score
        run_faithfulness  : whether to call the LLM faithfulness judge
                            (slower, costs an extra LLM call)
        llm               : optional pre-built LLM; created from settings if None
    """
    breakdown: dict = {}
    weighted_sum  = 0.0
    total_weight  = 0.0

    for name, fn, weight in _DETERMINISTIC_METRICS:
        try:
            val = fn(result)
        except Exception:
            val = 0.0
        breakdown[name] = round(val, 4)
        weighted_sum   += val * weight
        total_weight   += weight

    if run_faithfulness:
        faith = metric_faithfulness(result, llm=llm)
        breakdown["faithfulness"] = round(faith, 4)
        weighted_sum += faith * _FAITHFULNESS_WEIGHT
        total_weight += _FAITHFULNESS_WEIGHT
    else:
        breakdown["faithfulness"] = None   # not run

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    return EvalScore(overall=round(overall, 4), breakdown=breakdown)
