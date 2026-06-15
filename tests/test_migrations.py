"""Schema migration: an existing pre-lineage_id SQLite DB upgrades in place instead of
failing with ``no such column: bullets.lineage_id`` (reproduces the reported P1 break)."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from career_assistant.export import build_resume_document
from career_assistant.storage import repo
from career_assistant.storage.db import make_engine, make_session_factory


def _build_old_schema(engine):
    """Create a bullets/suggestions schema as it existed *before* lineage_id / integrity_flags."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE resumes (id VARCHAR PRIMARY KEY, user_id VARCHAR, "
                "filename VARCHAR, raw_text TEXT, parsed_json JSON, created_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE resume_versions (id VARCHAR PRIMARY KEY, resume_id VARCHAR, "
                "parent_version_id VARCHAR, version_type VARCHAR, label VARCHAR, "
                "created_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE bullets (id VARCHAR PRIMARY KEY, resume_version_id VARCHAR, "
                "section VARCHAR, order_index INTEGER, original_text TEXT, current_text TEXT, "
                "keywords_json JSON)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE tailoring_suggestions (id VARCHAR PRIMARY KEY, bullet_id VARCHAR, "
                "jd_id VARCHAR, suggested_text TEXT, reasoning TEXT, llm_judgment FLOAT, "
                "embedding_similarity FLOAT, integrity_score FLOAT, integrity_band VARCHAR, "
                "status VARCHAR, created_at DATETIME)"
            )
        )
        conn.execute(
            text("INSERT INTO resumes (id, filename, raw_text) VALUES ('r1', 'old.pdf', 'x')")
        )
        conn.execute(
            text(
                "INSERT INTO resume_versions (id, resume_id, version_type) "
                "VALUES ('v1', 'r1', 'original')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO bullets (id, resume_version_id, section, order_index, "
                "original_text, current_text) VALUES "
                "('b1', 'v1', 'experience', 0, 'Built X', 'Built X'),"
                "('b2', 'v1', 'experience', 1, 'Shipped Y', 'Shipped Y')"
            )
        )
        # A legacy pending suggestion (no integrity_flags_json column yet).
        conn.execute(
            text(
                "INSERT INTO tailoring_suggestions (id, bullet_id, jd_id, suggested_text, status) "
                "VALUES ('s1', 'b1', 'j1', 'Engineered X', 'pending')"
            )
        )


def _add_child_version(engine):
    """Add an accepted child v2 of v1 whose bullets are positional copies (old copy invariant)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO resume_versions (id, resume_id, parent_version_id, version_type) "
                "VALUES ('v2', 'r1', 'v1', 'accepted')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO bullets (id, resume_version_id, section, order_index, "
                "original_text, current_text) VALUES "
                "('b3', 'v2', 'experience', 0, 'Built X', 'Engineered X'),"
                "('b4', 'v2', 'experience', 1, 'Shipped Y', 'Shipped Y')"
            )
        )


def test_old_db_reads_fail_before_migration(tmp_path):
    # Confirm the reported break exists on the legacy schema (no lineage_id column).
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    from sqlalchemy.exc import OperationalError

    with make_session_factory(engine)() as s, pytest.raises(OperationalError, match="lineage_id"):
        repo.list_bullets(s, "v1")


def test_create_db_upgrades_old_schema_in_place(tmp_path):
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)

    repo.create_db(engine)  # create_all + run_migrations

    cols = {c["name"] for c in inspect(engine).get_columns("bullets")}
    assert "lineage_id" in cols
    sugg_cols = {c["name"] for c in inspect(engine).get_columns("tailoring_suggestions")}
    assert "integrity_flags_json" in sugg_cols

    with make_session_factory(engine)() as s:
        bullets = repo.list_bullets(s, "v1")  # would OperationalError without the migration
        assert [b.current_text for b in bullets] == ["Built X", "Shipped Y"]
        # Each legacy bullet self-roots with a distinct, non-null lineage.
        lineages = {b.lineage_id for b in bullets}
        assert all(lineages) and len(lineages) == 2
        # Phase 12 export (which reads lineage via the ORM) now works on the upgraded DB.
        doc = build_resume_document(s, version_id="v1")
        assert doc.extra_experience == ["Built X", "Shipped Y"]


def test_legacy_parent_child_share_lineage_by_position(tmp_path):
    # The crux of the improved backfill: a positionally-copied child bullet inherits the
    # parent bullet's lineage, so old accepted/customized chains (and their pending suggestions)
    # stay applicable to the head instead of being severed into independent ids.
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    _add_child_version(engine)

    repo.create_db(engine)

    lineage = _lineage_map(engine)
    assert lineage["b1"] == lineage["b3"]  # (experience, 0) carries across the copy
    assert lineage["b2"] == lineage["b4"]  # (experience, 1) too
    assert lineage["b1"] != lineage["b2"]  # distinct logical bullets keep distinct lineages


def test_old_suggestions_read_and_create_after_migration(tmp_path):
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    repo.create_db(engine)

    from career_assistant.storage.models import TailoringSuggestionRow

    with make_session_factory(engine)() as s:
        # The legacy suggestion reads back, with the new column defaulting to NULL.
        legacy = s.get(TailoringSuggestionRow, "s1")
        assert legacy.suggested_text == "Engineered X"
        assert legacy.integrity_flags_json is None
        # And a new suggestion persists its flags into the freshly added column.
        created = repo.create_suggestion(
            s,
            bullet_id="b1",
            jd_id="j1",
            suggested_text="Drove X",
            integrity_flags={"unsupported_claims": ["led a team"]},
        )
        s.commit()
        assert s.get(TailoringSuggestionRow, created.id).integrity_flags_json == {
            "unsupported_claims": ["led a team"]
        }


def test_unique_index_exists_after_migration(tmp_path):
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    repo.create_db(engine)

    indexes = inspect(engine).get_indexes("bullets")
    unique = next(i for i in indexes if i["name"] == "ux_bullets_resume_version_lineage_id")
    assert unique["unique"]
    assert unique["column_names"] == ["resume_version_id", "lineage_id"]


def test_partial_upgrade_is_repaired(tmp_path):
    # P1: a half-migrated DB — lineage_id column present but all NULL, no lineage indexes.
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    _add_child_version(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE bullets ADD COLUMN lineage_id VARCHAR"))  # column only
    assert all(v is None for v in _lineage_map(engine).values())  # left NULL by the prior run

    repo.create_db(engine)  # must backfill NULLs and create the missing indexes

    lineage = _lineage_map(engine)
    assert all(lineage.values())  # every row now has a lineage
    assert lineage["b1"] == lineage["b3"] and lineage["b2"] == lineage["b4"]  # positional link
    index_names = {i["name"] for i in inspect(engine).get_indexes("bullets")}
    assert "ux_bullets_resume_version_lineage_id" in index_names


def test_duplicate_positions_fail_loudly(tmp_path):
    # P2: a parent with two (experience, 0) bullets makes positional inheritance ambiguous.
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO bullets (id, resume_version_id, section, order_index, "
                "original_text, current_text) VALUES "
                "('dup', 'v1', 'experience', 0, 'Another at pos 0', 'Another at pos 0')"
            )
        )
    with pytest.raises(RuntimeError, match="ambiguous duplicate"):
        repo.create_db(engine)


def test_migration_is_idempotent(tmp_path):
    engine = make_engine(str(tmp_path / "old.db"))
    _build_old_schema(engine)
    repo.create_db(engine)
    lineage_before = _lineage_map(engine)

    repo.create_db(engine)  # second pass must not error or change anything
    assert _lineage_map(engine) == lineage_before


def test_create_db_on_fresh_db_is_unaffected(tmp_path):
    # The migration is a no-op when create_all already built the current schema.
    engine = make_engine(str(tmp_path / "fresh.db"))
    repo.create_db(engine)
    repo.create_db(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("bullets")}
    assert "lineage_id" in cols


def _lineage_map(engine) -> dict[str, str]:
    with engine.connect() as conn:
        return {row[0]: row[1] for row in conn.execute(text("SELECT id, lineage_id FROM bullets"))}
