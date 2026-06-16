"""Phase 0 cost/latency budget estimator.

Counts the LLM + embedding calls one ``(resume × JD)`` application makes across the
pipeline, prints an estimated $/run, and projects the 100-JD benchmark cost (the
gate before the Phase 16 full evaluation).

Run:  python -m career_assistant.eval.budget
      python -m career_assistant.eval.budget --jds 100 --tailored 12

The token-size assumptions below are deliberate, tunable guesses — adjust them as
real traffic lands. All pricing comes from ``career_assistant.llm.pricing``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from career_assistant.config import settings
from career_assistant.llm.pricing import (
    COMPLETION_PRICING,
    estimate_completion_cost,
    estimate_embedding_cost,
)

# --- Per-call token-size assumptions (averages) ------------------------------
# Completion calls: (prompt_tokens, completion_tokens)
RESUME_EXTRACT_TOKENS = (2500, 1200)  # full resume in, structured JSON out
JD_ANALYZE_TOKENS = (1500, 700)  # JD in, structured JSON out
FIT_SCORE_TOKENS = (1800, 500)  # parsed resume + JD in, breakdown out
TAILOR_TOKENS = (700, 250)  # one bullet + JD context in, rewrite out
INTEGRITY_TOKENS = (600, 200)  # tailored claim + source bullet in, judgment out

# Embedding calls: tokens per item
TOKENS_PER_BULLET = 40
TOKENS_PER_JD_CHUNK = 35
TOKENS_PER_TAILORED_BULLET = 45

# --- Rough latency assumptions (seconds per call), sequential upper bound ----
COMPLETION_LATENCY_S = 3.0
HOSTED_EMBED_LATENCY_S = 0.4  # one batched API round-trip
LOCAL_EMBED_LATENCY_S = 0.05  # local sentence-transformers, per item

_LOCAL_EMBEDDING_PROVIDERS = {"huggingface", "hf", "sentence-transformers", "sentence_transformers"}


def is_local_embedding(provider: str) -> bool:
    """Whether the embedding provider runs locally (free per token)."""
    return provider.lower().strip() in _LOCAL_EMBEDDING_PROVIDERS


def embedding_label(provider: str, model: str) -> str:
    """Human label for the embedding line ('… (local, $0)' for local providers)."""
    return f"{model} (local, $0)" if is_local_embedding(provider) else model


@dataclass
class ApplicationProfile:
    """Work done for one (resume × JD) application.

    ``resume_*`` items run once per resume; everything else runs per JD. The
    distinction matters because a resume is parsed/embedded once and reused
    across many JDs, so amortized cost drops as JDs per resume grows.
    """

    n_bullets: int = 20  # bullets extracted + embedded from the resume
    n_jd_chunks: int = 10  # JD requirement chunks embedded
    n_tailored: int = 8  # bullets actually tailored for this JD
    jds_per_resume: int = 1  # amortize one-time resume cost over this many JDs


@dataclass
class LineItem:
    label: str
    kind: str  # "completion" | "embedding"
    n_calls: int
    cost: float | None  # None => unknown price for this model
    latency_s: float
    once_per_resume: bool = False


@dataclass
class Summary:
    items: list[LineItem] = field(default_factory=list)

    @property
    def per_run_cost(self) -> float | None:
        # None if any priced item is unknown; otherwise the amortized total.
        total = 0.0
        for it in self.items:
            if it.cost is None:
                return None
            total += it.cost
        return total

    @property
    def per_run_latency_s(self) -> float:
        return sum(it.latency_s for it in self.items)


def build_summary(
    profile: ApplicationProfile,
    *,
    llm_model: str,
    embedding_model: str,
    embedding_is_local: bool,
) -> Summary:
    """Build the amortized per-application breakdown for the given profile."""
    amortize = max(profile.jds_per_resume, 1)

    def completion(label: str, n: int, tokens: tuple[int, int], *, once: bool) -> LineItem:
        per_call = estimate_completion_cost(llm_model, tokens[0], tokens[1])
        eff_n = n / amortize if once else n
        cost = None if per_call is None else per_call * eff_n
        return LineItem(label, "completion", n, cost, COMPLETION_LATENCY_S * eff_n, once)

    def embedding(label: str, n_items: int, tokens_per: int, *, once: bool) -> LineItem:
        eff_n = n_items / amortize if once else n_items
        if embedding_is_local:
            cost = 0.0
            latency = LOCAL_EMBED_LATENCY_S * eff_n
        else:
            cost = estimate_embedding_cost(embedding_model, int(tokens_per * eff_n))
            latency = HOSTED_EMBED_LATENCY_S  # single batched round-trip
        return LineItem(label, "embedding", n_items, cost, latency, once)

    return Summary(
        items=[
            completion("Resume extraction", 1, RESUME_EXTRACT_TOKENS, once=True),
            embedding("Resume bullet embeddings", profile.n_bullets, TOKENS_PER_BULLET, once=True),
            completion("JD analysis", 1, JD_ANALYZE_TOKENS, once=False),
            embedding("JD chunk embeddings", profile.n_jd_chunks, TOKENS_PER_JD_CHUNK, once=False),
            completion("Fit scoring", 1, FIT_SCORE_TOKENS, once=False),
            completion("Tailoring", profile.n_tailored, TAILOR_TOKENS, once=False),
            completion("Integrity judging", profile.n_tailored, INTEGRITY_TOKENS, once=False),
            embedding(
                "Tailored bullet embeddings",
                profile.n_tailored,
                TOKENS_PER_TAILORED_BULLET,
                once=False,
            ),
        ]
    )


def _fmt(cost: float | None) -> str:
    return "  (unknown price)" if cost is None else f"${cost:.4f}"


def render(summary: Summary, *, n_jds: int, llm_model: str, embedding_label: str) -> str:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("Phase 0 — Cost / Latency Budget")
    lines.append("=" * 74)
    lines.append(f"completion model : {llm_model}")
    lines.append(f"embeddings       : {embedding_label}")
    if llm_model not in COMPLETION_PRICING:
        lines.append(f"  !! no pricing for {llm_model!r} — add it to llm/pricing.py for $ totals")
    lines.append("-" * 74)
    lines.append(f"{'item':<38}{'kind':<12}{'calls':>6}{'cost':>10}{'lat(s)':>8}")
    lines.append("-" * 74)
    for it in summary.items:
        tag = " (1×/resume)" if it.once_per_resume else ""
        lines.append(
            f"{it.label + tag:<38}{it.kind:<12}{it.n_calls:>6}"
            f"{_fmt(it.cost):>10}{it.latency_s:>8.2f}"
        )
    lines.append("-" * 74)
    per_run = summary.per_run_cost
    lines.append(f"{'per application (1 resume × 1 JD)':<46}{_fmt(per_run):>10}")
    lines.append(f"{'  sequential latency upper bound':<46}{summary.per_run_latency_s:>8.2f}s")
    if per_run is not None:
        lines.append(f"{f'{n_jds}-JD benchmark':<46}{f'${per_run * n_jds:.2f}':>10}")
    else:
        lines.append(f"{n_jds}-JD benchmark           (unknown — set model pricing)")
    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 0 cost/latency budget estimator")
    parser.add_argument("--jds", type=int, default=100, help="benchmark size (default 100)")
    parser.add_argument("--bullets", type=int, default=20, help="resume bullets")
    parser.add_argument("--jd-chunks", type=int, default=10, help="JD requirement chunks")
    parser.add_argument("--tailored", type=int, default=8, help="bullets tailored per JD")
    parser.add_argument(
        "--jds-per-resume",
        type=int,
        default=1,
        help="amortize one-time resume parse/embed cost over this many JDs",
    )
    args = parser.parse_args(argv)

    profile = ApplicationProfile(
        n_bullets=args.bullets,
        n_jd_chunks=args.jd_chunks,
        n_tailored=args.tailored,
        jds_per_resume=args.jds_per_resume,
    )
    summary = build_summary(
        profile,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_is_local=is_local_embedding(settings.embedding_provider),
    )
    print(
        render(
            summary,
            n_jds=args.jds,
            llm_model=settings.llm_model,
            embedding_label=embedding_label(settings.embedding_provider, settings.embedding_model),
        )
    )


if __name__ == "__main__":
    main()
