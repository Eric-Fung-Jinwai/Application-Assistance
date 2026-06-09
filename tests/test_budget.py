"""Pure-Python checks for pricing + the Phase 0 budget estimator (no network)."""

from __future__ import annotations

from career_assistant.eval.budget import ApplicationProfile, build_summary
from career_assistant.llm.pricing import (
    estimate_completion_cost,
    estimate_embedding_cost,
    format_cost,
)


def test_completion_cost_known_model():
    # gpt-4o: $2.50/1M in, $10.00/1M out
    cost = estimate_completion_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50


def test_completion_cost_unknown_model_is_none():
    assert estimate_completion_cost("does-not-exist", 1000, 1000) is None


def test_embedding_cost_local_is_free():
    # Unknown / local models are treated as free.
    assert estimate_embedding_cost("sentence-transformers/all-MiniLM-L6-v2", 1_000_000) == 0.0


def test_embedding_cost_hosted():
    assert estimate_embedding_cost("text-embedding-3-small", 1_000_000) == 0.02


def test_format_cost():
    assert format_cost(None) == "?"
    assert format_cost(0.0) == "$0.000000"


def test_summary_priced_model_local_embeddings():
    profile = ApplicationProfile(n_bullets=20, n_jd_chunks=10, n_tailored=8)
    summary = build_summary(
        profile,
        llm_model="gpt-4o-mini",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_is_local=True,
    )
    # All items priced => per-run cost is a positive float, embeddings free.
    assert summary.per_run_cost is not None
    assert summary.per_run_cost > 0
    embed_cost = sum(it.cost for it in summary.items if it.kind == "embedding")
    assert embed_cost == 0.0
    assert summary.per_run_latency_s > 0


def test_summary_unknown_model_yields_none():
    profile = ApplicationProfile()
    summary = build_summary(
        profile,
        llm_model="mystery-model",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_is_local=True,
    )
    assert summary.per_run_cost is None


def test_amortization_reduces_cost():
    profile_one = ApplicationProfile(jds_per_resume=1)
    profile_many = ApplicationProfile(jds_per_resume=10)
    kw = dict(
        llm_model="gpt-4o-mini",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_is_local=True,
    )
    one = build_summary(profile_one, **kw).per_run_cost
    many = build_summary(profile_many, **kw).per_run_cost
    # Amortizing the one-time resume parse over more JDs lowers per-run cost.
    assert many < one
