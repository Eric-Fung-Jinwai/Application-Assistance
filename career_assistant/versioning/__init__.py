"""Resume version control: immutable snapshot tree + rollback/restore (Phase 9)."""

from career_assistant.versioning.versions import (
    bullets_by_lineage,
    create_child_version,
    find_original_version,
    get_version,
    latest_accepted_ancestor,
    latest_accepted_version,
    list_versions,
    restore_original,
    rollback_to,
    version_chain,
)

__all__ = [
    "bullets_by_lineage",
    "create_child_version",
    "find_original_version",
    "get_version",
    "latest_accepted_ancestor",
    "latest_accepted_version",
    "list_versions",
    "restore_original",
    "rollback_to",
    "version_chain",
]
