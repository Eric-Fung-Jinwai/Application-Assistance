"""Model pricing table + cost estimation (Phase 0 budget).

Prices are USD per 1M tokens. These are ESTIMATES — confirm against the
provider's current pricing page before trusting the benchmark numbers. Unknown
models return ``None`` so callers can surface "?" rather than silently logging
a $0.00 cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float  # USD per 1M prompt tokens
    output_per_1m: float  # USD per 1M completion tokens


# Completion-model pricing, keyed by the id used in settings.llm_model.
# NOTE: "gpt-5.4-nano" pricing is a placeholder estimate — verify before Phase 16.
COMPLETION_PRICING: dict[str, ModelPricing] = {
    "gpt-5.4-nano": ModelPricing(input_per_1m=0.2, output_per_1m=1.25),
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4o": ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
}

# Hosted-embedding pricing, USD per 1M tokens.
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Local sentence-transformers models cost nothing per token (compute only).
LOCAL_EMBEDDING_COST_PER_1M = 0.0


def estimate_completion_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """USD cost of one completion call, or None if the model price is unknown."""
    pricing = COMPLETION_PRICING.get(model)
    if pricing is None:
        return None
    return (
        prompt_tokens / 1_000_000 * pricing.input_per_1m
        + completion_tokens / 1_000_000 * pricing.output_per_1m
    )


def estimate_embedding_cost(model: str, total_tokens: int) -> float:
    """USD cost of an embedding call. Unknown / local models are treated as free."""
    per_1m = EMBEDDING_PRICING.get(model, LOCAL_EMBEDDING_COST_PER_1M)
    return total_tokens / 1_000_000 * per_1m


def format_cost(cost: float | None) -> str:
    """Render a cost for logs: '$0.001234' or '?' when unknown."""
    return "?" if cost is None else f"${cost:.6f}"
