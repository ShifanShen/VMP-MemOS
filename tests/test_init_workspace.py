"""Tests for the non-destructive workspace initializer."""

from scripts.init_workspace import LOG_FILES, WORKSPACE_DIRECTORIES, init_workspace


def test_init_workspace_creates_expected_layout(tmp_path) -> None:
    workspace = tmp_path / "memory_workspace"

    changed = init_workspace(workspace)

    assert changed
    assert (workspace / "INDEX.md").is_file()
    assert (workspace / "MEMORY.md").is_file()
    assert all((workspace / name).is_dir() for name in WORKSPACE_DIRECTORIES)
    assert all((workspace / "logs" / name).is_file() for name in LOG_FILES)


def test_init_workspace_is_idempotent_and_preserves_content(tmp_path) -> None:
    workspace = tmp_path / "memory_workspace"
    init_workspace(workspace)
    custom_content = "# User-maintained index\n"
    (workspace / "INDEX.md").write_text(custom_content, encoding="utf-8")
    (workspace / "logs" / "operations.jsonl").write_text('{"op":"ADD"}\n', encoding="utf-8")

    changed = init_workspace(workspace)

    assert changed == []
    assert (workspace / "INDEX.md").read_text(encoding="utf-8") == custom_content
    assert (workspace / "logs" / "operations.jsonl").read_text(encoding="utf-8") == (
        '{"op":"ADD"}\n'
    )

