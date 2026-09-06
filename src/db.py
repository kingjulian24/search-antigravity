"""Database schema, connection management, and full-text search operations."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "conversations.db"


def get_db_path() -> Path:
    """Return configured database path via env var or default."""
    env_path = os.environ.get("ANTIGRAVITY_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create and configure a SQLite connection with FTS5 and row factory."""
    target_path = Path(db_path) if db_path else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite tables, FTS5 virtual table, triggers, and indices."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                total_steps INTEGER DEFAULT 0,
                file_path TEXT,
                last_indexed_mtime REAL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                created_at TEXT,
                source TEXT,
                type TEXT,
                content TEXT,
                tool_names TEXT,
                has_thinking BOOLEAN DEFAULT 0,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv_step 
            ON messages(conversation_id, step_index);
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations(updated_at DESC);
        """)

        # FTS5 Virtual Table for full-text BM25 search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                tool_names,
                conversation_id UNINDEXED,
                step_index UNINDEXED,
                tokenize = 'porter unicode61'
            );
        """)

        # Synchronization Triggers for messages <-> messages_fts
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, tool_names, conversation_id, step_index)
                VALUES (new.id, new.content, new.tool_names, new.conversation_id, new.step_index);
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
                INSERT INTO messages_fts(rowid, content, tool_names, conversation_id, step_index)
                VALUES (new.id, new.content, new.tool_names, new.conversation_id, new.step_index);
            END;
        """)


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user search query for SQLite FTS5 syntax while preserving quotes and boolean operators."""
    query = query.strip()
    if not query:
        return ""

    # Clean unbalanced quotes
    if query.count('"') % 2 != 0:
        query = query.replace('"', ' ')

    pattern = re.compile(r'"([^"]*)"|(\S+)')
    tokens: List[str] = []

    for match in pattern.finditer(query):
        quoted, word = match.groups()
        if quoted is not None:
            clean_q = re.sub(r"[^\w\s\-]", " ", quoted).strip()
            if clean_q:
                tokens.append(f'"{clean_q}"')
        elif word is not None:
            upper = word.upper()
            if upper in ("AND", "OR", "NOT"):
                tokens.append(upper)
            elif word.endswith("*") and any(c.isalnum() for c in word[:-1]):
                clean_p = re.sub(r"[^\w\-]", "", word[:-1])
                if clean_p:
                    tokens.append(f"{clean_p}*")
            else:
                clean_w = re.sub(r"[^\w\-]", "", word).strip()
                if clean_w:
                    tokens.append(f'"{clean_w}"')

    return " ".join(tokens)



def search_messages(
    conn: sqlite3.Connection,
    query: str,
    conversation_id: Optional[str] = None,
    type_filter: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search messages using FTS5 BM25 relevance ranking."""
    fts_query = sanitize_fts5_query(query)
    if not fts_query:
        return []

    limit = max(1, min(limit, 50))
    where_clauses = ["messages_fts MATCH ?"]
    params: List[Any] = [fts_query]

    if conversation_id:
        where_clauses.append("m.conversation_id = ?")
        params.append(conversation_id)

    if type_filter:
        clean_type = type_filter.strip().upper()
        where_clauses.append("m.type = ?")
        params.append(clean_type)

    where_sql = " AND ".join(where_clauses)
    params.append(limit)

    sql = f"""
        SELECT
            m.id,
            m.conversation_id,
            m.step_index,
            m.created_at,
            m.source,
            m.type,
            m.tool_names,
            m.has_thinking,
            c.title,
            snippet(messages_fts, 0, '**', '**', '...', 35) AS snippet,
            bm25(messages_fts) AS rank_score
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {where_sql}
        ORDER BY rank_score ASC
        LIMIT ?
    """

    cursor = conn.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


def get_step_context(
    conn: sqlite3.Connection,
    conversation_id: str,
    step_index: int,
    context_window: int = 2,
) -> List[Dict[str, Any]]:
    """Retrieve dialogue messages surrounding a specific step index."""
    min_step = max(0, step_index - max(0, context_window))
    max_step = step_index + max(0, context_window)

    sql = """
        SELECT
            m.id,
            m.conversation_id,
            m.step_index,
            m.created_at,
            m.source,
            m.type,
            m.content,
            m.tool_names,
            m.has_thinking,
            c.title
        FROM messages m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.conversation_id = ?
          AND m.step_index >= ?
          AND m.step_index <= ?
        ORDER BY m.step_index ASC
    """
    cursor = conn.execute(sql, (conversation_id, min_step, max_step))
    return [dict(row) for row in cursor.fetchall()]


def list_recent_conversations(
    conn: sqlite3.Connection,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List recent conversations sorted by latest activity."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    sql = """
        SELECT
            conversation_id,
            title,
            created_at,
            updated_at,
            total_steps,
            file_path
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT ? OFFSET ?
    """
    cursor = conn.execute(sql, (limit, offset))
    return [dict(row) for row in cursor.fetchall()]


def get_database_stats(
    conn: sqlite3.Connection,
    db_path: Path | str | None = None,
) -> Dict[str, Any]:
    """Calculate and return repository metrics."""
    target_path = Path(db_path) if db_path else get_db_path()

    conv_row = conn.execute(
        "SELECT COUNT(*) AS total_convs, MIN(created_at) AS earliest, MAX(updated_at) AS latest FROM conversations"
    ).fetchone()

    msg_row = conn.execute("SELECT COUNT(*) AS total_messages FROM messages").fetchone()
    
    type_rows = conn.execute(
        "SELECT type, COUNT(*) AS count FROM messages GROUP BY type"
    ).fetchall()
    by_type = {row["type"]: row["count"] for row in type_rows}

    size_bytes = target_path.stat().st_size if target_path.exists() else 0

    return {
        "total_conversations": conv_row["total_convs"] if conv_row else 0,
        "total_messages": msg_row["total_messages"] if msg_row else 0,
        "earliest_date": conv_row["earliest"] if conv_row else None,
        "latest_date": conv_row["latest"] if conv_row else None,
        "messages_by_type": by_type,
        "db_size_bytes": size_bytes,
        "db_size_mb": round(size_bytes / (1024 * 1024), 2),
        "db_path": str(target_path),
    }
