"""Phase 16 benchmark labeling helper.

The fit benchmark is a set of hand-labeled ``resume × JD`` pairs — a real resume, a real JD, and
*your* honest fit band (the "match" is the label, not a historical hire). A focused single-role
set of ~20 pairs spanning strong/moderate/weak is a real regression gate (the harness checks a
*percentage*), so the scope is flexible, not a fixed 100-pair quota. Each pair is stored
**pre-parsed** (a full ``ParsedResume`` + ``ParsedJD``) so the harness needs no LLM at eval time.
Writing that JSON by hand is impractical, so this tool runs the **real parsers once** on raw
resume/JD text: a labeler only supplies the resume file, the JD file, the role, and the band.

The parse step is the *only* place this calls the LLM (the per-pair "budget" cost the item is
gated on). Evaluation via ``harness.py`` stays fully offline against the JSON it writes.

Commands::

    python -m career_assistant.eval.label add \\
        --name swe_strong_01 --role SWE --band strong \\
        --resume path/to/resume.txt --jd path/to/jd.txt
    python -m career_assistant.eval.label validate
    python -m career_assistant.eval.label stats

``add`` parses + appends one labeled pair; ``validate`` schema-checks the file; ``stats`` shows
progress toward the 25-each target. The label (expected band) is always **yours** — the tool
never guesses it, so no machine-made label can sneak into the regression gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from career_assistant.domain import ParsedJD, ParsedResume, RawResume
from career_assistant.ingest.extract import extract_resume
from career_assistant.jd import analyze_jd
from career_assistant.llm.client import LLMClient

DATASETS_DIR = Path(__file__).parent / "datasets"
BENCHMARK_PATH = DATASETS_DIR / "fit_pairs_benchmark.json"

# Suggested role tracks (not enforced — a single-role benchmark is a valid scope). Any
# non-empty role string is accepted; unknown ones just get a warning.
KNOWN_ROLES = ("AI-Eng", "AI-PM", "SWE", "DS")

# Default size target for a useful single-role benchmark. The harness gate is a *percentage*,
# so a smaller real set is a real regression gate — this is guidance, not a hard requirement.
BENCHMARK_TARGET = 20

# Valid fit bands (must mirror ``fit.engine._band``). These ARE enforced: they're the gate labels.
BANDS = ("strong", "moderate", "weak")


def load_benchmark(path: Path | None = None) -> list[dict]:
    """Load the benchmark pairs (empty list when the file doesn't exist yet).

    ``path`` defaults to ``BENCHMARK_PATH`` resolved *at call time*, so monkeypatching the
    module global in tests works (a default-arg binding would freeze the original path).
    """
    path = path or BENCHMARK_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_benchmark(pairs: list[dict], path: Path | None = None) -> None:
    (path or BENCHMARK_PATH).write_text(json.dumps(pairs, indent=2) + "\n")


def build_fit_pair(
    *,
    name: str,
    role: str,
    expected_band: str,
    resume_text: str,
    jd_text: str,
    llm: LLMClient | None = None,
) -> dict:
    """Parse raw resume + JD text into a pre-parsed, labeled benchmark entry.

    Validates ``role`` / ``expected_band`` up front so a typo can't quietly produce an
    un-scoreable pair. The returned dict matches ``fit_pairs.json``'s schema exactly.
    ``role`` is free-form metadata (any non-empty string); ``expected_band`` must be a real
    gate label.
    """
    if not role.strip():
        raise ValueError("role must be a non-empty string")
    if expected_band not in BANDS:
        raise ValueError(f"expected_band must be one of {BANDS}, got {expected_band!r}")
    if not resume_text.strip():
        raise ValueError("resume_text is empty")
    if not jd_text.strip():
        raise ValueError("jd_text is empty")

    raw = RawResume(filename=f"{name}.txt", raw_text=resume_text)
    resume = extract_resume(raw, llm=llm)
    jd = analyze_jd(jd_text, llm=llm)

    return {
        "name": name,
        "role": role,
        "expected_band": expected_band,
        "resume": resume.model_dump(mode="json"),
        "jd": jd.model_dump(mode="json"),
    }


def add_fit_pair(
    *,
    name: str,
    role: str,
    expected_band: str,
    resume_text: str,
    jd_text: str,
    llm: LLMClient | None = None,
    path: Path | None = None,
) -> dict:
    """Build a labeled pair and append it to the benchmark file (rejecting duplicate names)."""
    path = path or BENCHMARK_PATH
    pairs = load_benchmark(path)
    if any(p["name"] == name for p in pairs):
        raise ValueError(f"a pair named {name!r} already exists in {path.name}")
    entry = build_fit_pair(
        name=name,
        role=role,
        expected_band=expected_band,
        resume_text=resume_text,
        jd_text=jd_text,
        llm=llm,
    )
    pairs.append(entry)
    save_benchmark(pairs, path)
    return entry


def validate_dataset(pairs: list[dict]) -> list[str]:
    """Return a list of problems (empty == valid). Checks schema, enums, and unique names."""
    problems: list[str] = []
    seen: set[str] = set()
    required = {"name", "role", "expected_band", "resume", "jd"}
    for i, p in enumerate(pairs):
        missing = required - p.keys()
        if missing:
            problems.append(f"[{i}] missing keys: {sorted(missing)}")
            continue
        if p["name"] in seen:
            problems.append(f"[{i}] duplicate name: {p['name']!r}")
        seen.add(p["name"])
        if not str(p["role"]).strip():
            problems.append(f"[{p['name']}] role is empty")
        if p["expected_band"] not in BANDS:
            problems.append(f"[{p['name']}] invalid expected_band: {p['expected_band']!r}")
        # The pre-parsed payloads must round-trip into the domain models the harness loads.
        try:
            ParsedResume.model_validate(p["resume"])
        except Exception as e:  # noqa: BLE001 — surface any validation failure to the labeler
            problems.append(f"[{p['name']}] resume does not parse: {e}")
        try:
            ParsedJD.model_validate(p["jd"])
        except Exception as e:  # noqa: BLE001
            problems.append(f"[{p['name']}] jd does not parse: {e}")
    return problems


def role_counts(pairs: list[dict]) -> dict[str, int]:
    """Count pairs per role actually present (sorted by count, descending)."""
    counts: dict[str, int] = {}
    for p in pairs:
        role = str(p.get("role", "")) or "(none)"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def band_counts(pairs: list[dict]) -> dict[str, int]:
    """Count pairs per expected band — a benchmark wants a spread, not all one band."""
    return {band: sum(1 for p in pairs if p.get("expected_band") == band) for band in BANDS}


# --- CLI ----------------------------------------------------------------------------


def _cmd_add(args: argparse.Namespace) -> int:
    resume_text = Path(args.resume).read_text()
    jd_text = Path(args.jd).read_text()
    entry = add_fit_pair(
        name=args.name,
        role=args.role,
        expected_band=args.band,
        resume_text=resume_text,
        jd_text=jd_text,
    )
    n = len(load_benchmark())
    if args.role not in KNOWN_ROLES:
        print(f"  note: {args.role!r} is not a preset role {KNOWN_ROLES} (allowed — just a label)")
    print(f"added {entry['name']!r} ({entry['role']}, {entry['expected_band']}) — {n} pairs total")
    return 0


def _cmd_validate(_: argparse.Namespace) -> int:
    problems = validate_dataset(load_benchmark())
    if problems:
        print(f"INVALID ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("valid")
    return 0


def _cmd_stats(_: argparse.Namespace) -> int:
    pairs = load_benchmark()
    total = len(pairs)
    print(f"benchmark: {total}/{BENCHMARK_TARGET} pairs {'#' * total}")
    print("by role:")
    for role, n in role_counts(pairs).items():
        print(f"  {role:<10} {n:>3}")
    print("by band (aim for a spread):")
    for band, n in band_counts(pairs).items():
        print(f"  {band:<10} {n:>3}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 16 benchmark labeling helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="parse raw resume + JD text and append a labeled pair")
    p_add.add_argument("--name", required=True, help="unique id for the pair")
    p_add.add_argument(
        "--role", required=True, help=f"role label (free-form; presets: {', '.join(KNOWN_ROLES)})"
    )
    p_add.add_argument("--band", required=True, choices=BANDS, help="your expected fit band")
    p_add.add_argument("--resume", required=True, help="path to a resume text file")
    p_add.add_argument("--jd", required=True, help="path to a JD text file")
    p_add.set_defaults(func=_cmd_add)

    sub.add_parser("validate", help="schema-check the benchmark file").set_defaults(
        func=_cmd_validate
    )
    sub.add_parser("stats", help="progress toward the 25-each target").set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
