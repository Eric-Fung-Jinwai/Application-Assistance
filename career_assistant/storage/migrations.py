"""Lightweight, idempotent schema migrations for existing SQLite DBs (pre-Alembic).

``Base.metadata.create_all`` only ever *creates missing tables* — it never alters an existing
one. So columns added after a database was first created (``bullets.lineage_id`` in Phase 9,
``tailoring_suggestions.integrity_flags_json`` in Phase 8) are absent on older files, and even a
plain read fails with ``OperationalError: no such column``. Until Phase 17 brings Alembic, this
module brings an existing schema forward.

It is built from **independently-guarded idempotent steps** rather than one all-or-nothing
branch, so a *partially* upgraded DB (e.g. the column was added by an earlier interrupted run
but values are still NULL or the indexes are missing) is repaired too — each step checks its own
precondition: ensure the column exists, backfill NULL lineages, guard duplicates, ensure indexes.
Every step is a no-op once already satisfied, so it is safe to run on every open.

Lineage backfill reconstructs the old positional copy invariant: the old version code copied
bullets at the same ``(section, order_index)``, so a root bullet self-roots with a fresh UUID and
a child bullet inherits the lineage of the parent bullet at its position. Only when no safe parent
match exists (a true root, a dangling parent, drifted structure) does a row self-root. This keeps
legacy accepted/customized chains — and the pending suggestions targeting them — applicable to the
head rather than severing them. Ambiguous legacy data (duplicate positions, or a backfill that
would put two bullets on one lineage within a version) **fails loudly** rather than mis-linking.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from sqlalchemy import Connection, Engine, inspect, text

# Named so the indexes are recognisable and the "ensure" steps stay idempotent.
_UNIQUE_INDEX = "ux_bullets_resume_version_lineage_id"
_LINEAGE_INDEX = "ix_bullets_lineage_id"
_UNIQUE_COLUMNS = ["resume_version_id", "lineage_id"]


def run_migrations(engine: Engine) -> None:
    """Bring an existing SQLite schema up to the current ORM (no-op on a fresh/empty DB)."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "bullets" in tables:
        _migrate_bullets(engine)
    if "tailoring_suggestions" in tables and "integrity_flags_json" not in _columns(
        inspector, "tailoring_suggestions"
    ):
        _add_suggestion_flags(engine)


def _columns(inspector, table: str) -> set[str]:
    return {col["name"] for col in inspector.get_columns(table)}


def _migrate_bullets(engine: Engine) -> None:
    """Repair the ``bullets`` lineage schema via independent, self-guarding steps.

    Each step is conditional on the live state, so a fresh DB, a fully-migrated DB, and a
    half-migrated DB (column present but NULL / indexes missing) all converge to the same result.
    """
    with engine.begin() as conn:
        _ensure_lineage_column(conn)
        if _has_null_lineage(conn):
            _guard_duplicate_positions(conn)  # positional inheritance must be unambiguous
            _backfill_null_lineage(conn)
        _guard_no_duplicate_lineage(conn)
        _ensure_lineage_indexes(conn)


def _ensure_lineage_column(conn: Connection) -> None:
    cols = {row[1] for row in conn.execute(text("PRAGMA table_info(bullets)"))}
    if "lineage_id" not in cols:
        # SQLite can't ADD COLUMN with a non-constant default → add nullable, then backfill.
        conn.execute(text("ALTER TABLE bullets ADD COLUMN lineage_id VARCHAR"))


def _has_null_lineage(conn: Connection) -> bool:
    row = conn.execute(text("SELECT 1 FROM bullets WHERE lineage_id IS NULL LIMIT 1")).first()
    return row is not None


def _guard_duplicate_positions(conn: Connection) -> None:
    """Fail loudly if any version has two bullets at one ``(section, order_index)``.

    Backfill matches a child bullet to its parent by position, so a duplicate position would make
    that match ambiguous (silently inheriting one parent's lineage arbitrarily). Refuse instead.
    """
    dupes = conn.execute(
        text(
            "SELECT resume_version_id, section, order_index, COUNT(*) AS c FROM bullets "
            "GROUP BY resume_version_id, section, order_index HAVING c > 1"
        )
    ).fetchall()
    if dupes:
        pairs = [(row[0], row[1], row[2]) for row in dupes]
        raise RuntimeError(
            "lineage migration cannot run: ambiguous duplicate (resume_version_id, section, "
            f"order_index) bullets make positional inheritance unsafe: {pairs}. Resolve these "
            "before upgrading."
        )


def _backfill_null_lineage(conn: Connection) -> None:
    """Fill only NULL ``lineage_id``s, inheriting from the parent version by position, else root.

    Existing (non-NULL) lineages are read first and left untouched, so a half-migrated DB keeps
    whatever it already assigned and a child can still inherit from a parent that was set earlier.
    Versions are processed parent-first so a child sees its parent's lineage by the time it runs.
    """
    parent_of = {
        vid: pid
        for vid, pid in conn.execute(text("SELECT id, parent_version_id FROM resume_versions"))
    }
    version_ids = set(parent_of)

    bullets_by_version: dict[str, list[tuple]] = defaultdict(list)
    lineage_of: dict[str, str] = {}
    for bid, vid, section, order_index, lineage in conn.execute(
        text("SELECT id, resume_version_id, section, order_index, lineage_id FROM bullets")
    ):
        bullets_by_version[vid].append((bid, section, order_index))
        if lineage is not None:
            lineage_of[bid] = lineage
    # (section, order_index) → bullet id, per version (positions are unique, guarded above).
    position_index = {
        vid: {(section, order_index): bid for bid, section, order_index in rows}
        for vid, rows in bullets_by_version.items()
    }

    to_write: dict[str, str] = {}
    done_versions: set[str] = set()
    for vid in _parent_first_order(version_ids, parent_of):
        parent = parent_of.get(vid)
        parent_ready = bool(parent) and parent in done_versions
        parent_positions = position_index.get(parent, {}) if parent else {}
        for bid, section, order_index in bullets_by_version.get(vid, []):
            if bid in lineage_of:
                continue  # already has a lineage — leave it
            inherited = None
            if parent_ready:
                parent_bid = parent_positions.get((section, order_index))
                inherited = lineage_of.get(parent_bid) if parent_bid else None
            lineage = inherited or str(uuid.uuid4())
            lineage_of[bid] = lineage
            to_write[bid] = lineage
        done_versions.add(vid)

    # Bullets whose version row is missing entirely (dangling) can't match a parent → self-root.
    for vid, rows in bullets_by_version.items():
        if vid not in version_ids:
            for bid, _section, _order in rows:
                if bid not in lineage_of:
                    lineage_of[bid] = to_write[bid] = str(uuid.uuid4())

    for bid, lineage in to_write.items():
        conn.execute(
            text("UPDATE bullets SET lineage_id = :lid WHERE id = :id"),
            {"lid": lineage, "id": bid},
        )


def _parent_first_order(version_ids: set[str], parent_of: dict[str, str | None]) -> list[str]:
    """Topological order: a version appears after its parent (roots/dangling-parents first)."""
    order: list[str] = []
    placed: set[str] = set()
    remaining = set(version_ids)
    while remaining:
        progressed = False
        for vid in list(remaining):
            parent = parent_of.get(vid)
            if not parent or parent not in version_ids or parent in placed:
                order.append(vid)
                placed.add(vid)
                remaining.discard(vid)
                progressed = True
        if not progressed:  # cycle (shouldn't happen) → place the rest; they'll self-root
            order.extend(remaining)
            break
    return order


def _guard_no_duplicate_lineage(conn: Connection) -> None:
    """Fail loudly if two bullets share a lineage within one version (before the unique index)."""
    dupes = conn.execute(
        text(
            "SELECT resume_version_id, lineage_id, COUNT(*) AS c FROM bullets "
            "WHERE lineage_id IS NOT NULL "
            "GROUP BY resume_version_id, lineage_id HAVING c > 1"
        )
    ).fetchall()
    if dupes:
        pairs = [(row[0], row[1]) for row in dupes]
        raise RuntimeError(
            "lineage migration produced duplicate (resume_version_id, lineage_id) pairs from "
            f"ambiguous legacy data: {pairs}."
        )


def _ensure_lineage_indexes(conn: Connection) -> None:
    """Create the lineage indexes if absent (repairs a half-migrated DB; skips existing ones)."""
    inspector = inspect(conn)
    indexes = inspector.get_indexes("bullets")
    has_lineage = any(ix["column_names"] == ["lineage_id"] for ix in indexes)
    has_unique = any(
        ix["unique"] and ix["column_names"] == _UNIQUE_COLUMNS for ix in indexes
    ) or any(
        uc["column_names"] == _UNIQUE_COLUMNS for uc in inspector.get_unique_constraints("bullets")
    )
    if not has_lineage:
        conn.execute(text(f"CREATE INDEX {_LINEAGE_INDEX} ON bullets (lineage_id)"))
    if not has_unique:
        conn.execute(
            text(f"CREATE UNIQUE INDEX {_UNIQUE_INDEX} ON bullets (resume_version_id, lineage_id)")
        )


def _add_suggestion_flags(engine: Engine) -> None:
    """Add the nullable ``integrity_flags_json`` column to ``tailoring_suggestions``."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tailoring_suggestions ADD COLUMN integrity_flags_json JSON"))
