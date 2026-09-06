"""Unit and integration tests for search-antigravity MCP server."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest

from src.db import (
    get_database_stats,
    get_db_connection,
    get_step_context,
    init_db,
    list_recent_conversations,
    sanitize_fts5_query,
    search_messages,
)
from src.indexer import (
    extract_tool_names,
    find_transcript_files,
    parse_transcript_file,
    sanitize_text_content,
    sync_index,
)
from src.server import (
    get_antigravity_stats,
    get_antigravity_step,
    list_antigravity_conversations,
    search_antigravity_conversations,
    sync_antigravity_index,
)


@pytest.fixture
def temp_workspace() -> Generator[Tuple[Path, Path], None, None]:
    """Provide a temporary brain directory and database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir(parents=True)
        db_path = tmp_path / "test_conversations.db"
        yield brain_dir, db_path


def create_synthetic_transcript(
    brain_dir: Path,
    conversation_id: str,
    steps: list[dict],
) -> Path:
    """Helper to write a synthetic transcript.jsonl file."""
    log_dir = brain_dir / conversation_id / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = log_dir / "transcript.jsonl"

    with open(transcript_file, "w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")

    return transcript_file


def test_sanitize_fts5_query() -> None:
    """Verify FTS5 query sanitizer handles special characters and booleans."""
    assert sanitize_fts5_query("hello world") == '"hello" "world"'
    assert sanitize_fts5_query('"exact phrase"') == '"exact phrase"'
    assert sanitize_fts5_query('sqlite AND "fts5 index"') == '"sqlite" AND "fts5 index"'
    assert sanitize_fts5_query("error OR bug NOT warning") == '"error" OR "bug" NOT "warning"'
    assert sanitize_fts5_query("prefix* search") == 'prefix* "search"'
    assert sanitize_fts5_query('unclosed "quote test') == '"unclosed" "quote" "test"'
    assert sanitize_fts5_query("punctuation!@#$%^&*()") == '"punctuation"'
    assert sanitize_fts5_query("   ") == ""


def test_extract_tool_names() -> None:
    """Verify tool call names extraction."""
    assert extract_tool_names(None) == ""
    assert extract_tool_names([]) == ""
    assert extract_tool_names([{"name": "view_file"}, {"name": "run_command"}]) == "view_file, run_command"
    # Deduplication test
    assert extract_tool_names([{"name": "view_file"}, {"name": "view_file"}]) == "view_file"


def test_sanitize_text_content() -> None:
    """Verify content truncation and normalization."""
    short_text = "Simple short prompt"
    assert sanitize_text_content(short_text) == short_text

    long_text = "x" * 15_000
    sanitized = sanitize_text_content(long_text)
    assert len(sanitized) < 11_000
    assert "[... truncated for search index ...]" in sanitized


def test_db_init_and_triggers(temp_workspace: tuple[Path, Path]) -> None:
    """Verify table creation and automatic FTS5 triggers."""
    _, db_path = temp_workspace
    conn = get_db_connection(db_path)
    init_db(conn)

    # Insert a conversation and message
    with conn:
        conn.execute("""
            INSERT INTO conversations (conversation_id, title, created_at, updated_at, total_steps, file_path, last_indexed_mtime)
            VALUES ('c1', 'Test Title', '2026-09-06T10:00:00Z', '2026-09-06T10:05:00Z', 1, '/tmp/c1', 12345.0)
        """)
        conn.execute("""
            INSERT INTO messages (conversation_id, step_index, created_at, source, type, content, tool_names, has_thinking)
            VALUES ('c1', 0, '2026-09-06T10:00:00Z', 'USER_EXPLICIT', 'USER_INPUT', 'How to configure FastMCP with SQLite FTS5?', '', 0)
        """)

    # Verify message was mirrored into messages_fts automatically via trigger
    fts_rows = conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'FastMCP'").fetchall()
    assert len(fts_rows) == 1
    assert fts_rows[0]["step_index"] == 0

    # Test update trigger
    with conn:
        conn.execute("UPDATE messages SET content = 'Updated BM25 search ranking query' WHERE id = 1")
    assert len(conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'FastMCP'").fetchall()) == 0
    assert len(conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'BM25'").fetchall()) == 1

    # Test delete trigger
    with conn:
        conn.execute("DELETE FROM messages WHERE id = 1")
    assert len(conn.execute("SELECT * FROM messages_fts WHERE messages_fts MATCH 'BM25'").fetchall()) == 0

    conn.close()


def test_incremental_indexing(temp_workspace: tuple[Path, Path]) -> None:
    """Verify initial indexing and incremental mtime-based updates."""
    brain_dir, db_path = temp_workspace

    steps_1 = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T12:00:00Z",
            "content": "# Refactor Database Layer\nPlease migrate our SQLite queries to use FTS5 full text search.",
        },
        {
            "step_index": 1,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-06T12:01:00Z",
            "content": "I will update the database schema and add the FTS5 virtual table.",
            "thinking": "Need to check SQLite version first.",
            "tool_calls": [{"name": "write_to_file", "args": {"TargetFile": "src/db.py"}}],
        },
    ]

    steps_2 = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T13:00:00Z",
            "content": "Deploying FastMCP server over stdio transport.",
        }
    ]

    t1 = create_synthetic_transcript(brain_dir, "conv-001", steps_1)
    t2 = create_synthetic_transcript(brain_dir, "conv-002", steps_2)

    # Initial scan
    res1 = sync_index(brain_dir=brain_dir, db_path=db_path)
    assert res1["discovered_transcripts"] == 2
    assert res1["indexed_new"] == 2
    assert res1["skipped"] == 0
    assert res1["total_messages"] == 3

    # Second scan without changes: should skip all
    res2 = sync_index(brain_dir=brain_dir, db_path=db_path)
    assert res2["discovered_transcripts"] == 2
    assert res2["indexed_new"] == 0
    assert res2["skipped"] == 2

    # Modify conv-001: add another step and touch mtime
    time.sleep(0.05)
    steps_1_updated = list(steps_1) + [
        {
            "step_index": 2,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T12:05:00Z",
            "content": "Check indexing latency and BM25 rank score accuracy.",
        }
    ]
    create_synthetic_transcript(brain_dir, "conv-001", steps_1_updated)
    # Ensure mtime is strictly greater
    os.utime(t1, (time.time() + 2, time.time() + 2))

    res3 = sync_index(brain_dir=brain_dir, db_path=db_path)
    assert res3["updated"] == 1
    assert res3["skipped"] == 1
    assert res3["total_messages"] == 3  # re-indexed all 3 steps of conv-001


def test_search_and_context(temp_workspace: tuple[Path, Path]) -> None:
    """Verify search ranking, filtering, and context retrieval."""
    brain_dir, db_path = temp_workspace

    steps = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T14:00:00Z",
            "content": "Can you implement a FastMCP server in Python?",
        },
        {
            "step_index": 1,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-06T14:00:30Z",
            "content": "Yes, FastMCP makes creating MCP servers with stdio transport straightforward.",
            "tool_calls": [{"name": "write_to_file"}],
        },
        {
            "step_index": 2,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T14:01:00Z",
            "content": "Now add SQLite FTS5 search index ranking with BM25.",
        },
    ]
    create_synthetic_transcript(brain_dir, "conv-alpha", steps)
    sync_index(brain_dir=brain_dir, db_path=db_path)

    conn = get_db_connection(db_path)
    try:
        # Search for FastMCP
        matches = search_messages(conn, "FastMCP")
        assert len(matches) == 2
        assert any(m["step_index"] == 0 for m in matches)
        assert any(m["step_index"] == 1 for m in matches)

        # Search with type filter
        user_matches = search_messages(conn, "FastMCP", type_filter="USER_INPUT")
        assert len(user_matches) == 1
        assert user_matches[0]["type"] == "USER_INPUT"

        # Search with conversation filter
        conv_matches = search_messages(conn, "BM25", conversation_id="conv-alpha")
        assert len(conv_matches) == 1
        assert conv_matches[0]["step_index"] == 2

        # Context retrieval
        context = get_step_context(conn, conversation_id="conv-alpha", step_index=1, context_window=1)
        assert len(context) == 3
        assert [c["step_index"] for c in context] == [0, 1, 2]

        # Stats
        stats = get_database_stats(conn, db_path)
        assert stats["total_conversations"] == 1
        assert stats["total_messages"] == 3
        assert stats["messages_by_type"]["USER_INPUT"] == 2
        assert stats["messages_by_type"]["PLANNER_RESPONSE"] == 1
    finally:
        conn.close()


def test_server_tools(monkeypatch: pytest.MonkeyPatch, temp_workspace: tuple[Path, Path]) -> None:
    """Verify formatting of exposed MCP server tools."""
    brain_dir, db_path = temp_workspace
    monkeypatch.setenv("ANTIGRAVITY_BRAIN_DIR", str(brain_dir))
    monkeypatch.setenv("ANTIGRAVITY_DB_PATH", str(db_path))

    steps = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-09-06T15:00:00Z",
            "content": "Diagnose latency spike in indexing pipeline.",
        },
        {
            "step_index": 1,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-06T15:01:00Z",
            "content": "Analyzing profiling data. SQLite WAL mode provides high throughput.",
            "tool_calls": [{"name": "run_command"}],
        },
    ]
    create_synthetic_transcript(brain_dir, "conv-perf", steps)

    # 1. sync_antigravity_index tool
    sync_out = sync_antigravity_index()
    assert "Index Sync Completed" in sync_out
    assert "New Conversations Indexed:** 1" in sync_out


    # 2. search_antigravity_conversations tool
    search_out = search_antigravity_conversations("latency spike")
    assert "Search Results for: 'latency spike'" in search_out
    assert "conv-perf" in search_out
    assert "Step #0" in search_out

    # 3. get_antigravity_step tool
    step_out = get_antigravity_step(conversation_id="conv-perf", step_index=1, context_window=1)
    assert "Conversation Context: `conv-perf`" in step_out
    assert "Step #1: MODEL (PLANNER_RESPONSE)" in step_out
    assert "[TARGET STEP]" in step_out
    assert "SQLite WAL mode" in step_out

    # 4. list_antigravity_conversations tool
    list_out = list_antigravity_conversations()
    assert "Recent Antigravity Conversations" in list_out
    assert "Diagnose latency spike in indexing pipeline" in list_out

    # 5. get_antigravity_stats tool
    stats_out = get_antigravity_stats()
    assert "Antigravity Index Statistics" in stats_out
    assert "Total Conversations:** 1" in stats_out
    assert "Total Messages:** 2" in stats_out
