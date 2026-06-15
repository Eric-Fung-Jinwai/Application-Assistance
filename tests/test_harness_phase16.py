"""Phase 16 acceptance: the harness prints a precision/recall table and a fit table, and
a regression that lets a planted fabrication through fails the run."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from career_assistant.config import Settings
from career_assistant.domain import IntegrityBand
from career_assistant.eval.harness import (
    FIT_ACCURACY_MIN,
    evaluate_fit,
    evaluate_integrity,
    fit_accuracy,
    load_fabrications,
    load_fit_pairs,
    main,
    sweep_integrity,
)
from career_assistant.tailor.integrity import band_for_score, compute_integrity_score

# --- fit-band evaluation ---------------------------------------------------------


def test_fit_eval_matches_labels():
    rows = evaluate_fit(load_fit_pairs())
    assert len(rows) == 10
    assert fit_accuracy(rows) == 1.0  # the current engine agrees with every label
    for r in rows:
        assert r.predicted == r.expected
    assert fit_accuracy(rows) >= FIT_ACCURACY_MIN


# --- integrity scoring primitives ------------------------------------------------


def test_planted_fabrication_scores_below_50_and_is_high_risk():
    # "led team of 10" with no team evidence — the canonical Phase 8 example.
    case = next(c for c in load_fabrications() if c["name"] == "led_team_of_10")
    score = compute_integrity_score(case["llm_judgment"], case["embedding_similarity"])
    assert score < 50
    assert band_for_score(score) == IntegrityBand.high_risk


def test_band_for_score_cutoffs():
    assert band_for_score(95) == IntegrityBand.safe
    assert band_for_score(75) == IntegrityBand.moderate
    assert band_for_score(55) == IntegrityBand.aggressive
    assert band_for_score(20) == IntegrityBand.high_risk


# --- integrity input/config validation -------------------------------------------


def test_compute_integrity_score_rejects_out_of_range_signals():
    with pytest.raises(ValueError, match="llm_judgment"):
        compute_integrity_score(1.5, 0.5)
    with pytest.raises(ValueError, match="embedding_similarity"):
        compute_integrity_score(0.5, -0.1)


def test_compute_integrity_score_rejects_unnormalized_weights():
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_integrity_score(0.5, 0.5, w_llm=0.5, w_embed=0.9)


def test_band_for_score_rejects_inverted_cutoffs():
    with pytest.raises(ValueError, match="aggressive <= moderate <= safe"):
        band_for_score(60, safe=90, moderate=70, aggressive=80)


def test_settings_reject_bad_integrity_config():
    # Weights that don't sum to 1 and inverted band cutoffs are caught at load.
    with pytest.raises(ValidationError):
        Settings(integrity_w_llm=0.5, integrity_w_embed=0.9)
    with pytest.raises(ValidationError):
        Settings(integrity_band_aggressive=80, integrity_band_moderate=70, integrity_band_safe=90)


# --- integrity precision/recall --------------------------------------------------


def test_integrity_eval_perfect_on_dataset():
    m = evaluate_integrity(load_fabrications())
    assert m.recall == 1.0  # all fabrications caught
    assert m.precision == 1.0  # no faithful rewrite flagged
    assert m.missed == []
    assert m.false_alarms == []


def test_regression_lets_fabrication_through_drops_recall():
    # Simulate a tailoring/judge regression: a real fabrication now gets a high grounding
    # judgment, so it is NOT flagged → recall falls below 1.0 and the gate must trip.
    cases = [
        {
            "name": "sneaky_fab",
            "label": "fabrication",
            "llm_judgment": 0.95,
            "embedding_similarity": 0.9,
        },
        {"name": "ok", "label": "faithful", "llm_judgment": 0.9, "embedding_similarity": 0.85},
    ]
    m = evaluate_integrity(cases)
    assert m.recall < 1.0
    assert "sneaky_fab" in m.missed


# --- sweep -----------------------------------------------------------------------


def test_sweep_finds_a_perfect_setting():
    best = sweep_integrity(load_fabrications())
    assert best.metrics.recall == 1.0
    assert best.metrics.precision == 1.0
    assert 0.0 < best.w_llm < 1.0
    assert abs(best.w_llm + best.w_embed - 1.0) < 1e-9


# --- CLI gate --------------------------------------------------------------------


def test_main_passes_on_clean_datasets(capsys):
    code = main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out
    assert "Fit bands" in out
    assert "Integrity" in out
