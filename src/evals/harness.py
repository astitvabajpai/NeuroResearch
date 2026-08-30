"""
Eval harness — versioned benchmark runner.

Produces files like:
    data/eval_results/2026-07-29_llama3-8b_standard.json

Usage (CLI):
    python -m src.evals.harness                         # all 25 fixtures
    python -m src.evals.harness --topic "transformers"  # one ad-hoc topic
    python -m src.evals.harness --from-db               # score saved DB sessions
    python -m src.evals.harness --faithfulness          # include LLM faithfulness judge
    python -m src.evals.harness --model Qwen/Qwen2.5-7B-Instruct  # model override

Usage (Python):
    from src.evals.harness import EvalSuite
    suite = EvalSuite(run_faithfulness=True)
    report = suite.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.evals.dataset import EVAL_DATASET, EvalFixture
from src.evals.metrics import EvalScore, ResearchResult, score_result

logger = logging.getLogger(__name__)


# ── Per-fixture result ────────────────────────────────────────────────────────

@dataclass
class FixtureResult:
    fixture: EvalFixture
    result: Optional[ResearchResult]
    eval_score: Optional[EvalScore]
    assertion_failures: List[str] = field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.error and not self.assertion_failures

    def short_summary(self) -> str:
        status  = "PASS" if self.passed else "FAIL"
        overall = f"{self.eval_score.overall:.3f}" if self.eval_score else "N/A"
        faith   = ""
        if self.eval_score and self.eval_score.breakdown.get("faithfulness") is not None:
            faith = f"  faith={self.eval_score.breakdown['faithfulness']:.2f}"
        return (
            f"[{status}] [{self.fixture.difficulty.upper():<11}] "
            f"{self.fixture.topic[:52]:<52} "
            f"overall={overall}{faith}  {self.elapsed_seconds:.1f}s"
        )


# ── Assertion checker ─────────────────────────────────────────────────────────

def check_assertions(result: ResearchResult, fixture: EvalFixture) -> List[str]:
    failures = []
    wc = len(result.draft.split())
    if wc < fixture.min_words:
        failures.append(f"word_count={wc} < {fixture.min_words}")

    draft_lower = result.draft.lower()
    for term in fixture.must_contain:
        if term.lower() not in draft_lower:
            failures.append(f"must_contain '{term}' missing from draft")

    headings = {
        h.strip().lower()
        for h in re.findall(r"^##\s+(.+)$", result.draft, re.MULTILINE)
    }
    for section in fixture.required_sections:
        if not any(section in h for h in headings):
            failures.append(f"required section '{section}' missing")

    if result.critique_score < fixture.min_score:
        failures.append(
            f"critique_score={result.critique_score:.3f} < {fixture.min_score}"
        )
    return failures


# ── Eval report ───────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    fixture_results: List[FixtureResult] = field(default_factory=list)
    model_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    run_faithfulness: bool = False

    def summary(self) -> str:
        total  = len(self.fixture_results)
        passed = sum(1 for r in self.fixture_results if r.passed)
        scored = [r for r in self.fixture_results if r.eval_score]
        avg_overall  = sum(r.eval_score.overall for r in scored) / max(1, len(scored))
        avg_faith    = [r.eval_score.breakdown.get("faithfulness") or 0
                        for r in scored
                        if r.eval_score and r.eval_score.breakdown.get("faithfulness") is not None]
        avg_faith_s  = f"  avg_faithfulness={sum(avg_faith)/len(avg_faith):.3f}" if avg_faith else ""
        avg_ret_prec = sum(r.eval_score.breakdown.get("retrieval_precision", 0) for r in scored) / max(1, len(scored))
        avg_ret_rec  = sum(r.eval_score.breakdown.get("retrieval_recall", 0) for r in scored) / max(1, len(scored))

        # Group by difficulty
        by_diff: dict = {}
        for fr in self.fixture_results:
            d = fr.fixture.difficulty
            by_diff.setdefault(d, []).append(fr)

        lines = [
            "",
            "=" * 80,
            f"  NeuroResearch Eval Report",
            f"  Model   : {self.model_id or 'default'}",
            f"  Started : {self.started_at}",
            "=" * 80,
        ]
        for fr in self.fixture_results:
            lines.append(f"  {fr.short_summary()}")
            for af in fr.assertion_failures:
                lines.append(f"        WARNING  {af}")
            if fr.error:
                lines.append(f"        ERROR    {fr.error[:120]}")

        lines += ["", "-" * 80, "  SUMMARY BY DIFFICULTY"]
        for diff in ["easy", "medium", "hard", "adversarial"]:
            items = by_diff.get(diff, [])
            if items:
                p = sum(1 for x in items if x.passed)
                avg = sum(x.eval_score.overall for x in items if x.eval_score) / max(1, len(items))
                lines.append(f"    {diff:<12}: {p}/{len(items)} passed  avg_overall={avg:.3f}")

        lines += [
            "-" * 80,
            f"  OVERALL: {passed}/{total} passed  avg_overall={avg_overall:.3f}"
            f"  ret_prec={avg_ret_prec:.3f}  ret_rec={avg_ret_rec:.3f}{avg_faith_s}",
            f"  Finished: {self.finished_at}",
            "=" * 80,
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        scored = [r for r in self.fixture_results if r.eval_score]
        return {
            "meta": {
                "model_id":          self.model_id,
                "started_at":        self.started_at,
                "finished_at":       self.finished_at,
                "run_faithfulness":  self.run_faithfulness,
                "total":             len(self.fixture_results),
                "passed":            sum(1 for r in self.fixture_results if r.passed),
                "avg_overall":       round(sum(r.eval_score.overall for r in scored) / max(1, len(scored)), 4),
                "avg_retrieval_precision": round(
                    sum(r.eval_score.breakdown.get("retrieval_precision", 0) for r in scored) / max(1, len(scored)), 4),
                "avg_retrieval_recall": round(
                    sum(r.eval_score.breakdown.get("retrieval_recall", 0) for r in scored) / max(1, len(scored)), 4),
            },
            "results": [
                {
                    "topic":              fr.fixture.topic,
                    "difficulty":         fr.fixture.difficulty,
                    "deep_research":      fr.fixture.deep_research,
                    "passed":             fr.passed,
                    "overall":            fr.eval_score.overall if fr.eval_score else None,
                    "breakdown":          fr.eval_score.breakdown if fr.eval_score else {},
                    "critique_score":     fr.result.critique_score if fr.result else None,
                    "iterations":         fr.result.iterations if fr.result else None,
                    "elapsed_seconds":    fr.elapsed_seconds,
                    "assertion_failures": fr.assertion_failures,
                    "error":              fr.error,
                }
                for fr in self.fixture_results
            ],
        }


# ── Eval suite ────────────────────────────────────────────────────────────────

class EvalSuite:
    """
    Runs the research pipeline against the benchmark dataset.

    Args:
        fixtures         : list of EvalFixture (defaults to full EVAL_DATASET)
        max_iterations   : max pipeline iterations per topic
        output_dir       : where to save versioned JSON reports
        model_id         : model to use for all agents (default from settings)
        research_model   : override research agent model
        writer_model     : override writer agent model
        critique_model   : override critique agent model
        run_faithfulness : call LLM judge for each result (slower but richer)
        difficulty_filter: only run fixtures with this difficulty tag
    """

    def __init__(
        self,
        fixtures: Optional[List[EvalFixture]] = None,
        max_iterations: int = 3,
        output_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        research_model: Optional[str] = None,
        writer_model: Optional[str] = None,
        critique_model: Optional[str] = None,
        run_faithfulness: bool = False,
        difficulty_filter: Optional[str] = None,
    ) -> None:
        all_fixtures = fixtures or EVAL_DATASET
        if difficulty_filter:
            all_fixtures = [f for f in all_fixtures if f.difficulty == difficulty_filter]
        self.fixtures          = all_fixtures
        self.max_iterations    = max_iterations
        self.model_id          = model_id
        self.research_model    = research_model or model_id
        self.writer_model      = writer_model   or model_id
        self.critique_model    = critique_model or model_id
        self.run_faithfulness  = run_faithfulness

        if output_dir is None:
            try:
                from src.config.settings import get_settings
                output_dir = get_settings().EVAL_OUTPUT_DIR
            except Exception:
                output_dir = "data/eval_results"
        self.output_dir = Path(output_dir)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, save: bool = True) -> EvalReport:
        report = EvalReport(
            model_id         = self.model_id or "default",
            started_at       = datetime.now(timezone.utc).isoformat(),
            run_faithfulness = self.run_faithfulness,
        )
        total = len(self.fixtures)
        for i, fixture in enumerate(self.fixtures, 1):
            print(f"  [{i}/{total}] {fixture.topic[:60]}...", flush=True)
            fr = self._run_fixture(fixture)
            report.fixture_results.append(fr)
            print(f"        {fr.short_summary()}", flush=True)

        report.finished_at = datetime.now(timezone.utc).isoformat()
        print(report.summary())
        if save:
            self._save_report(report)
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_fixture(self, fixture: EvalFixture) -> FixtureResult:
        start = time.time()
        try:
            from src.main import run_research
            raw = run_research(
                topic=fixture.topic,
                max_iterations=self.max_iterations,
                research_model=self.research_model,
                writer_model=self.writer_model,
                critique_model=self.critique_model,
                deep_research=fixture.deep_research,
            )
            draft = raw.get("draft", "")
            if hasattr(draft, "content"):
                draft = draft.content

            result = ResearchResult(
                topic          = fixture.topic,
                draft          = str(draft),
                research_notes = raw.get("research_notes", []),
                critique_score = float(raw.get("critique_score", 0.0)),
                iterations     = int(raw.get("iteration", 0)),
                deep_research  = fixture.deep_research,
                gold_facts     = fixture.gold_facts,
                gold_sources   = fixture.gold_sources,
            )
            eval_score = score_result(
                result,
                run_faithfulness=self.run_faithfulness,
            )
            failures = check_assertions(result, fixture)

            return FixtureResult(
                fixture            = fixture,
                result             = result,
                eval_score         = eval_score,
                assertion_failures = failures,
                elapsed_seconds    = time.time() - start,
            )

        except Exception as exc:
            logger.exception("Fixture '%s' raised: %s", fixture.topic, exc)
            return FixtureResult(
                fixture         = fixture,
                result          = None,
                eval_score      = None,
                error           = str(exc),
                elapsed_seconds = time.time() - start,
            )

    def _save_report(self, report: EvalReport) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Slugify model name
        model_slug = re.sub(r"[^a-zA-Z0-9-]", "-",
                            (self.model_id or "default").split("/")[-1])[:20]
        mode_slug  = "deep" if any(f.deep_research for f in self.fixtures) else "standard"
        filename   = f"{date_str}_{model_slug}_{mode_slug}.json"
        path       = self.output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[Eval] Report saved → {path}")
        return path


# ── Score saved DB sessions ───────────────────────────────────────────────────

def score_db_sessions(limit: int = 50, run_faithfulness: bool = False) -> List[dict]:
    """Load recent sessions from SQLite and score them offline."""
    from src.database import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, topic, draft, critique_score, iterations, research_notes "
            "FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    scored = []
    for row in rows:
        notes  = json.loads(row["research_notes"]) if row["research_notes"] else []
        result = ResearchResult(
            topic          = row["topic"],
            draft          = row["draft"],
            research_notes = notes,
            critique_score = float(row["critique_score"]),
            iterations     = int(row["iterations"]),
            session_id     = row["id"],
        )
        es = score_result(result, run_faithfulness=run_faithfulness)
        entry = {
            "session_id":    row["id"],
            "topic":         row["topic"],
            "critique_score": result.critique_score,
            "overall":       es.overall,
            **{k: v for k, v in es.breakdown.items() if v is not None},
        }
        scored.append(entry)
        faith_str = f"  faith={es.breakdown['faithfulness']:.2f}" \
                    if es.breakdown.get("faithfulness") is not None else ""
        print(f"  {row['id'][:8]}  {row['topic'][:48]:<48}  "
              f"overall={es.overall:.3f}{faith_str}")
    return scored


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s — %(message)s")
    p = argparse.ArgumentParser(description="NeuroResearch eval harness")
    p.add_argument("--topic",       default="",    help="Run a single ad-hoc topic")
    p.add_argument("--from-db",     action="store_true")
    p.add_argument("--deep",        action="store_true")
    p.add_argument("--faithfulness",action="store_true", help="Run LLM faithfulness judge")
    p.add_argument("--no-save",     action="store_true")
    p.add_argument("--difficulty",  default="",    help="Filter by difficulty tag")
    p.add_argument("--model",       default=None,  help="Model ID for all agents")
    p.add_argument("--research-model", default=None)
    p.add_argument("--writer-model",   default=None)
    p.add_argument("--critique-model", default=None)
    args = p.parse_args()

    if args.from_db:
        print("Scoring sessions from database...")
        score_db_sessions(run_faithfulness=args.faithfulness)
        return

    if args.topic:
        fixtures = [EvalFixture(topic=args.topic, deep_research=args.deep)]
    elif args.difficulty:
        fixtures = None   # will be filtered inside EvalSuite
    else:
        fixtures = None

    suite = EvalSuite(
        fixtures          = fixtures,
        model_id          = args.model,
        research_model    = args.research_model,
        writer_model      = args.writer_model,
        critique_model    = args.critique_model,
        run_faithfulness  = args.faithfulness,
        difficulty_filter = args.difficulty or None,
    )
    suite.run(save=not args.no_save)


if __name__ == "__main__":
    _main()
