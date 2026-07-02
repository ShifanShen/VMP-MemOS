#!/usr/bin/env python3
"""Create the default VMP-MemOS workspace without overwriting user data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
WORKSPACE_DIRECTORIES: Final = (
    "memories",
    "projects",
    "skills",
    "episodes",
    "resources",
    "archive",
    "versions",
    "vector",
    "cache",
    "logs",
)
LOG_FILES: Final = (
    "operations.jsonl",
    "retrievals.jsonl",
    "evaluations.jsonl",
)
SEED_FILES: Final = {
    "INDEX.md": """# Memory Workspace Index

This index will contain links to active project, skill, episode, and resource memories.
""",
    "MEMORY.md": """# Core Memory

No core memories have been stored yet.
""",
}


def init_workspace(workspace: Path, *, force: bool = False) -> list[Path]:
    """Initialize ``workspace`` and return paths created or refreshed.

    The function is idempotent. Existing Markdown content and JSONL logs are
    preserved unless ``force`` is used, and even then log files are never truncated.
    """

    changed: list[Path] = []
    workspace.mkdir(parents=True, exist_ok=True)

    for directory_name in WORKSPACE_DIRECTORIES:
        directory = workspace / directory_name
        if not directory.exists():
            directory.mkdir(parents=True)
            changed.append(directory)

    for filename, content in SEED_FILES.items():
        path = workspace / filename
        if force or not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")
            changed.append(path)

    for filename in LOG_FILES:
        path = workspace / "logs" / filename
        if not path.exists():
            path.touch()
            changed.append(path)

    return changed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "memory_workspace",
        help="Workspace path (default: <project>/memory_workspace).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh INDEX.md and MEMORY.md; JSONL logs are still preserved.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    changed = init_workspace(workspace, force=args.force)

    print(f"Workspace ready: {workspace}")
    if changed:
        print(f"Created or refreshed {len(changed)} path(s).")
    else:
        print("No changes required; existing workspace data was preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
