"""Phase 3 acceptance: parse PDF/DOCX/TXT into structured sections; image PDF errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from career_assistant.domain import ParsedResume, RawResume
from career_assistant.ingest import extract_resume, parse_file
from career_assistant.ingest.errors import (
    EmptyResumeError,
    ScannedPDFError,
    UnsupportedResumeError,
)
from career_assistant.llm.client import LLMClient

SAMPLE = (
    "Jane Doe — Software Engineer\n"
    "Experience: Built a Python data pipeline that cut latency by 30%.\n"
    "Led migration of services to Docker and Kubernetes.\n"
    "Skills: Python, SQL, Docker. Education: BS Computer Science."
)


# --- helpers ---------------------------------------------------------------------


def _make_text_pdf(path: Path, lines: list[str]) -> None:
    """Build a minimal valid one-page text PDF (no third-party generator needed)."""
    content = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for line in lines:
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content += f"({esc}) Tj T*\n"
    content += "ET"
    # Standard-14 fonts are single-byte; replace anything outside latin-1.
    content_b = content.encode("latin-1", errors="replace")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(content_b)).encode()
        + b" >>\nstream\n"
        + content_b
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    n = len(objs) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n" + f"<< /Size {n} /Root 1 0 R >>\n".encode()
    out += b"startxref\n" + f"{xref_pos}\n".encode() + b"%%EOF"
    path.write_bytes(out)


class FakeLLM(LLMClient):
    """Returns a canned structured extraction without hitting the network."""

    def __init__(self, payload: dict | str) -> None:
        self.payload = payload

    def complete(self, system, user, *, json_schema=None, temperature=0.2):
        return self.payload


_GOOD_PAYLOAD = {
    "education": [{"institution": "State U", "degree": "BS", "field_of_study": "CS"}],
    "experience": [
        {
            "title": "Software Engineer",
            "company": "Acme",
            "bullets": ["Built a Python data pipeline that cut latency by 30%."],
        }
    ],
    "projects": [],
    "skills": ["Python", "SQL", "Docker"],
    "certifications": [],
    "confidence": 0.95,
}


# --- raw-text extraction (PDF / DOCX / TXT) --------------------------------------


def test_parse_txt(tmp_path):
    f = tmp_path / "resume.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    raw = parse_file(str(f))
    assert isinstance(raw, RawResume)
    assert "Python data pipeline" in raw.raw_text
    assert raw.content_type == ".txt"


def test_parse_docx(tmp_path):
    f = tmp_path / "resume.docx"
    doc = DocxDocument()
    for line in SAMPLE.splitlines():
        doc.add_paragraph(line)
    doc.save(str(f))
    raw = parse_file(str(f))
    assert "Kubernetes" in raw.raw_text


def test_parse_pdf(tmp_path):
    f = tmp_path / "resume.pdf"
    _make_text_pdf(f, SAMPLE.splitlines())
    raw = parse_file(str(f))
    assert "Jane Doe" in raw.raw_text


def test_image_only_pdf_raises(tmp_path):
    # A blank page = pages exist but no extractable text → scanned/image PDF.
    f = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with f.open("wb") as fh:
        writer.write(fh)
    with pytest.raises(ScannedPDFError):
        parse_file(str(f))


def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(EmptyResumeError):
        parse_file(str(f))


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "resume.rtf"
    f.write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(UnsupportedResumeError):
        parse_file(str(f))


def test_corrupt_pdf_raises_typed(tmp_path):
    # Garbage bytes with a .pdf extension: pypdf raises its own error; we must map it.
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4 this is not really a pdf " + b"\x00\x01\x02" * 40)
    with pytest.raises(UnsupportedResumeError):
        parse_file(str(f))


def test_corrupt_docx_raises_typed(tmp_path):
    # Not a valid zip/docx package: python-docx raises PackageNotFoundError.
    f = tmp_path / "bad.docx"
    f.write_bytes(b"not a real docx file, just text pretending" * 3)
    with pytest.raises(UnsupportedResumeError):
        parse_file(str(f))


def test_binary_garbage_txt_rejected(tmp_path):
    # 80 NUL bytes decode to 80 chars (clears the length gate) but aren't text.
    f = tmp_path / "garbage.txt"
    f.write_bytes(b"\x00" * 80)
    with pytest.raises(EmptyResumeError):
        parse_file(str(f))


# --- LLM structured extraction ---------------------------------------------------


def test_extract_resume_structured():
    raw = RawResume(filename="r.txt", raw_text=SAMPLE)
    parsed = extract_resume(raw, llm=FakeLLM(_GOOD_PAYLOAD))
    assert isinstance(parsed, ParsedResume)
    assert parsed.skills == ["Python", "SQL", "Docker"]
    assert parsed.experience[0].company == "Acme"
    assert parsed.education[0].degree == "BS"
    assert parsed.confidence == 0.95


def test_extract_salvages_partial_on_bad_field():
    # experience is malformed (not a list of objects) → dropped; rest survives.
    payload = {**_GOOD_PAYLOAD, "experience": "not a list", "confidence": 0.9}
    raw = RawResume(filename="r.txt", raw_text=SAMPLE)
    parsed = extract_resume(raw, llm=FakeLLM(payload))
    assert parsed.skills == ["Python", "SQL", "Docker"]  # kept
    assert parsed.experience == []  # dropped, defaulted
    assert parsed.confidence == 0.5  # lowered, don't crash


def test_extract_non_json_string_is_safe():
    raw = RawResume(filename="r.txt", raw_text=SAMPLE)
    parsed = extract_resume(raw, llm=FakeLLM("totally not json"))
    assert parsed.confidence == 0.0
    assert parsed.skills == []
