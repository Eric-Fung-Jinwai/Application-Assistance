"""Phase 16 benchmark labeling helper: parses raw resume/JD text into pre-parsed labeled
pairs, validates the file, and tracks progress — without ever machine-guessing a label."""

from __future__ import annotations

import json

import pytest

from career_assistant.domain import ParsedJD, ParsedResume
from career_assistant.eval import label
from career_assistant.eval.harness import evaluate_fit, fit_accuracy
from career_assistant.llm.client import LLMClient


class _FixedParserLLM(LLMClient):
    """Returns a resume extraction for the parser prompt and a JD analysis otherwise, so the
    real ``extract_resume`` / ``analyze_jd`` pipelines run end-to-end with no network."""

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        if "resume parser" in system:
            return {
                "experience": [
                    {
                        "title": "ML Engineer",
                        "company": "Acme",
                        "bullets": ["Built and shipped recommendation models in Python"],
                    }
                ],
                "skills": ["Python", "PyTorch", "ML"],
                "education": [],
                "projects": [],
                "certifications": [],
                "confidence": 0.9,
            }
        # JD analysis payload.
        return {
            "title": "ML Engineer",
            "required_skills": ["Python", "ML"],
            "preferred_skills": ["PyTorch"],
            "responsibilities": ["Build ML models"],
        }


def test_build_fit_pair_parses_into_domain_models():
    entry = label.build_fit_pair(
        name="aieng_strong_01",
        role="AI-Eng",
        expected_band="strong",
        resume_text="ML Engineer at Acme. Built recommendation models in Python.",
        jd_text="ML Engineer. Required: Python, ML.",
        llm=_FixedParserLLM(),
    )
    assert entry["name"] == "aieng_strong_01"
    assert entry["role"] == "AI-Eng"
    assert entry["expected_band"] == "strong"
    # The stored payloads round-trip into the same models the harness loads.
    ParsedResume.model_validate(entry["resume"])
    ParsedJD.model_validate(entry["jd"])


def test_build_fit_pair_rejects_empty_role_and_bad_band():
    with pytest.raises(ValueError, match="role must be a non-empty string"):
        label.build_fit_pair(
            name="x",
            role="   ",
            expected_band="strong",
            resume_text="r",
            jd_text="j",
            llm=_FixedParserLLM(),
        )
    with pytest.raises(ValueError, match="expected_band must be one of"):
        label.build_fit_pair(
            name="x",
            role="SWE",
            expected_band="great",
            resume_text="r",
            jd_text="j",
            llm=_FixedParserLLM(),
        )


def test_build_fit_pair_rejects_empty_text():
    with pytest.raises(ValueError, match="resume_text is empty"):
        label.build_fit_pair(
            name="x",
            role="SWE",
            expected_band="strong",
            resume_text="   ",
            jd_text="j",
            llm=_FixedParserLLM(),
        )


def test_add_fit_pair_appends_and_rejects_duplicate(tmp_path):
    path = tmp_path / "bench.json"
    label.add_fit_pair(
        name="dup",
        role="DS",
        expected_band="moderate",
        resume_text="data scientist",
        jd_text="DS role",
        llm=_FixedParserLLM(),
        path=path,
    )
    assert len(label.load_benchmark(path)) == 1
    with pytest.raises(ValueError, match="already exists"):
        label.add_fit_pair(
            name="dup",
            role="DS",
            expected_band="weak",
            resume_text="x",
            jd_text="y",
            llm=_FixedParserLLM(),
            path=path,
        )


def test_validate_dataset_flags_problems():
    pairs = [
        {
            "name": "ok",
            "role": "SWE",
            "expected_band": "strong",
            "resume": ParsedResume().model_dump(mode="json"),
            "jd": ParsedJD().model_dump(mode="json"),
        },
        {"name": "ok", "role": "", "expected_band": "nope", "resume": {}, "jd": {}},
        {"name": "missing_keys"},
    ]
    problems = label.validate_dataset(pairs)
    assert any("duplicate name" in p for p in problems)
    assert any("role is empty" in p for p in problems)
    assert any("invalid expected_band" in p for p in problems)
    assert any("missing keys" in p for p in problems)


def test_validate_dataset_passes_on_real_built_pair(tmp_path):
    path = tmp_path / "bench.json"
    label.add_fit_pair(
        name="aipm_01",
        role="AI-PM",
        expected_band="moderate",
        resume_text="Product manager for ML",
        jd_text="AI PM role",
        llm=_FixedParserLLM(),
        path=path,
    )
    assert label.validate_dataset(label.load_benchmark(path)) == []


def test_role_counts_only_lists_present_roles():
    pairs = [
        {"name": "a", "role": "SWE"},
        {"name": "b", "role": "SWE"},
        {"name": "c", "role": "DS"},
    ]
    counts = label.role_counts(pairs)
    assert counts == {"SWE": 2, "DS": 1}  # only roles actually present, sorted by count


def test_build_fit_pair_accepts_free_form_role():
    entry = label.build_fit_pair(
        name="custom_role_01",
        role="Platform Eng",  # not a preset — allowed
        expected_band="moderate",
        resume_text="Platform engineer",
        jd_text="Platform role",
        llm=_FixedParserLLM(),
    )
    assert entry["role"] == "Platform Eng"


def test_labeled_pair_is_scoreable_by_harness(tmp_path):
    # The whole point: a labeled pair drops straight into the offline fit eval.
    path = tmp_path / "bench.json"
    label.add_fit_pair(
        name="aieng_strong_02",
        role="AI-Eng",
        expected_band="strong",
        resume_text="ML Engineer. Built ML models in Python with PyTorch.",
        jd_text="ML Engineer. Required: Python, ML.",
        llm=_FixedParserLLM(),
        path=path,
    )
    rows = evaluate_fit(label.load_benchmark(path))
    assert len(rows) == 1
    assert 0.0 <= fit_accuracy(rows) <= 1.0  # runs without error; band is computed offline


def test_stats_cli_reports_progress(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bench.json"
    monkeypatch.setattr(label, "BENCHMARK_PATH", path)
    label.add_fit_pair(
        name="swe_01",
        role="SWE",
        expected_band="strong",
        resume_text="Software engineer",
        jd_text="SWE role",
        llm=_FixedParserLLM(),
        path=path,
    )
    assert label.main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "SWE" in out
    assert f"/{label.BENCHMARK_TARGET}" in out


def test_validate_cli_round_trips_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps([]) + "\n")
    monkeypatch.setattr(label, "BENCHMARK_PATH", path)
    assert label.main(["validate"]) == 0
    assert "valid" in capsys.readouterr().out
