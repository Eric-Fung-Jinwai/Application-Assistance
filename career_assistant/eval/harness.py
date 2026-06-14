"""Phase 16 evaluation harness.

Two checks, both deterministic and offline so they can gate every prompt/code change:

  * **Fit bands** — run the Phase 6 ``score_fit`` over hand-labeled ``resume × JD`` pairs
    (stored *pre-parsed*, so no LLM call) and compare the predicted band to the label.
  * **Integrity precision/recall** — over planted fabrications + faithful rewrites, decide
    whether each is *flagged* (lands in the ``high_risk`` band) and score precision/recall
    against the labels. The judge's ``(llm_judgment, embedding_similarity)`` outputs are
    **cached in the dataset**, so the formula + cutoffs (Phase 8 / config) can be swept
    offline without re-invoking the LLM.

The harness **fails** (non-zero exit) when fit accuracy drops below the floor or *any*
planted fabrication slips through — e.g. a tailoring-prompt regression that the integrity
judge can no longer catch.

Run:  python -m career_assistant.eval.harness
      python -m career_assistant.eval.harness --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from career_assistant.domain import IntegrityBand, ParsedJD, ParsedResume
from career_assistant.fit import score_fit
from career_assistant.llm.client import EmbeddingClient
from career_assistant.tailor.integrity import band_for_score, compute_integrity_score

DATASETS_DIR = Path(__file__).parent / "datasets"

# Regression gates.
FIT_ACCURACY_MIN = 0.8  # fraction of pairs whose predicted band must match the label
INTEGRITY_MIN_RECALL = 1.0  # every planted fabrication must be caught

# A claim is "flagged" (blocked/warned) when its score lands in the high-risk band.
FLAGGED_BAND = IntegrityBand.high_risk

FABRICATION_LABEL = "fabrication"


# --- Fit-band evaluation ------------------------------------------------------------


@dataclass
class FitRow:
    name: str
    role: str
    expected: str
    predicted: str
    overall: float

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted


def load_fit_pairs() -> list[dict]:
    return json.loads((DATASETS_DIR / "fit_pairs.json").read_text())


def evaluate_fit(pairs: list[dict], *, embedder: EmbeddingClient | None = None) -> list[FitRow]:
    """Score each labeled pair and record predicted vs expected band."""
    rows: list[FitRow] = []
    for p in pairs:
        resume = ParsedResume.model_validate(p["resume"])
        jd = ParsedJD.model_validate(p["jd"])
        fit = score_fit(resume, jd, embedder=embedder)
        rows.append(
            FitRow(
                name=p["name"],
                role=p.get("role", ""),
                expected=p["expected_band"],
                predicted=fit.band or "",
                overall=fit.overall,
            )
        )
    return rows


def fit_accuracy(rows: list[FitRow]) -> float:
    return sum(r.correct for r in rows) / len(rows) if rows else 1.0


# --- Integrity precision/recall -----------------------------------------------------


@dataclass
class IntegrityMetrics:
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    missed: list[str]  # fabrications that slipped through (the dangerous failure)
    false_alarms: list[str]  # faithful rewrites wrongly flagged


def load_fabrications() -> list[dict]:
    return json.loads((DATASETS_DIR / "fabrications.json").read_text())


def _is_flagged(
    case: dict,
    *,
    w_llm: float | None,
    w_embed: float | None,
    aggressive: int | None,
) -> bool:
    score = compute_integrity_score(
        case["llm_judgment"], case["embedding_similarity"], w_llm=w_llm, w_embed=w_embed
    )
    return band_for_score(score, aggressive=aggressive) == FLAGGED_BAND


def evaluate_integrity(
    cases: list[dict],
    *,
    w_llm: float | None = None,
    w_embed: float | None = None,
    aggressive: int | None = None,
) -> IntegrityMetrics:
    """Precision/recall of 'flag as fabrication' against the labels.

    Positive class = a planted fabrication. ``recall`` is the share of fabrications
    caught (the metric the regression gate watches); ``precision`` is how many flagged
    items were truly fabrications (faithful rewrites flagged are false alarms).
    """
    tp = fp = fn = 0
    missed: list[str] = []
    false_alarms: list[str] = []
    for case in cases:
        is_fabrication = case["label"] == FABRICATION_LABEL
        flagged = _is_flagged(case, w_llm=w_llm, w_embed=w_embed, aggressive=aggressive)
        if is_fabrication and flagged:
            tp += 1
        elif is_fabrication and not flagged:
            fn += 1
            missed.append(case["name"])
        elif not is_fabrication and flagged:
            fp += 1
            false_alarms.append(case["name"])

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return IntegrityMetrics(precision, recall, tp, fp, fn, missed, false_alarms)


# --- Weight / cutoff sweep ----------------------------------------------------------


@dataclass
class SweepResult:
    w_llm: float
    w_embed: float
    aggressive: int
    metrics: IntegrityMetrics


# Candidate grids (kept small + explicit). w_embed = 1 - w_llm so the score stays 0–100.
SWEEP_W_LLM = (0.6, 0.7, 0.8, 0.9)
SWEEP_AGGRESSIVE = (40, 50, 60)


def sweep_integrity(cases: list[dict]) -> SweepResult:
    """Grid-search weights + the high-risk cutoff, maximizing recall then precision.

    Returns the best setting; the caller decides whether to write it to ``config``
    (this harness only recommends — it never edits config behind your back).
    """
    best: SweepResult | None = None
    for w_llm in SWEEP_W_LLM:
        w_embed = round(1.0 - w_llm, 3)
        for aggressive in SWEEP_AGGRESSIVE:
            m = evaluate_integrity(cases, w_llm=w_llm, w_embed=w_embed, aggressive=aggressive)
            cand = SweepResult(w_llm, w_embed, aggressive, m)
            if best is None or (m.recall, m.precision) > (
                best.metrics.recall,
                best.metrics.precision,
            ):
                best = cand
    assert best is not None  # grids are non-empty
    return best


# --- CLI ----------------------------------------------------------------------------


def _print_fit(rows: list[FitRow]) -> float:
    acc = fit_accuracy(rows)
    print("\n== Fit bands ==")
    print(f"{'pair':<26}{'role':<9}{'expected':<10}{'predicted':<10}{'overall':>8}  ok")
    for r in rows:
        print(
            f"{r.name:<26}{r.role:<9}{r.expected:<10}{r.predicted:<10}"
            f"{r.overall:>8.1f}  {'✓' if r.correct else '✗'}"
        )
    print(f"accuracy: {acc:.0%} ({sum(r.correct for r in rows)}/{len(rows)})")
    return acc


def _print_integrity(m: IntegrityMetrics) -> None:
    print("\n== Integrity (planted fabrications) ==")
    print(f"precision: {m.precision:.0%}   recall: {m.recall:.0%}")
    print(f"tp={m.true_positives} fp={m.false_positives} fn={m.false_negatives}")
    if m.missed:
        print(f"MISSED fabrications: {', '.join(m.missed)}")
    if m.false_alarms:
        print(f"false alarms: {', '.join(m.false_alarms)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 16 evaluation harness")
    parser.add_argument("--sweep", action="store_true", help="grid-search integrity weights/cutoff")
    args = parser.parse_args(argv)

    fit_rows = evaluate_fit(load_fit_pairs())
    acc = _print_fit(fit_rows)

    fabrications = load_fabrications()
    metrics = evaluate_integrity(fabrications)
    _print_integrity(metrics)

    if args.sweep:
        best = sweep_integrity(fabrications)
        print("\n== Sweep recommendation ==")
        print(
            f"INTEGRITY_W_LLM={best.w_llm} INTEGRITY_W_EMBED={best.w_embed} "
            f"band_aggressive={best.aggressive} "
            f"→ recall {best.metrics.recall:.0%}, precision {best.metrics.precision:.0%}"
        )
        print("(set these in .env / config.Settings to apply — not written automatically)")

    # Regression gates.
    failures: list[str] = []
    if acc < FIT_ACCURACY_MIN:
        failures.append(f"fit accuracy {acc:.0%} < {FIT_ACCURACY_MIN:.0%}")
    if metrics.recall < INTEGRITY_MIN_RECALL:
        failures.append(f"integrity recall {metrics.recall:.0%} < {INTEGRITY_MIN_RECALL:.0%}")

    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
