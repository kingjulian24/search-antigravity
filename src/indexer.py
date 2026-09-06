"""Scanner and parser for Antigravity conversation transcripts."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.db import get_db_connection, init_db

logger = logging.getLogger("search_antigravity.indexer")

DEFAULT_BRAIN_DIR = Path.home() / ".gemini" / "antigravity" / "brain"
MAX_CONTENT_CHARS = 10_000  # Truncate overly long content to avoid bloating FTS


def get_brain_dir() -> Path:
    """Return the configured brain directory."""
    env_dir = os.environ.get("ANTIGRAVITY_BRAIN_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_BRAIN_DIR


def extract_tool_names(tool_calls: Any) -> str:
    """Extract comma-separated tool names from a list of tool call objects."""
    if not isinstance(tool_calls, list):
        return ""
    names = []
    for call in tool_calls:
        if isinstance(call, dict) and "name" in call:
            name = str(call["name"]).strip()
            if name and name not in names:
                names.append(name)
    return ", ".join(names)


def sanitize_text_content(content: Any) -> str:
    """Normalize and truncate message content for indexing."""
    if content is None:
        return ""
    if not isinstance(content, str):
        content = str(content)
    
    content = content.strip()
    if len(content) > MAX_CONTENT_CHARS:
        return content[:MAX_CONTENT_CHARS] + "\n[... truncated for search index ...]"
    return content


def parse_transcript_file(transcript_path: Path, conversation_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse a transcript.jsonl file into conversation metadata and message records."""
    messages: List[Dict[str, Any]] = []
    earliest_time: Optional[str] = None
    latest_time: Optional[str] = None
    derived_title: Optional[str] = None

    file_mtime = transcript_path.stat().st_mtime
    file_dt_str = datetime.fromtimestamp(file_mtime, tz=timezone.utc).isoformat()

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                if not isinstance(record, dict):
                    continue

                step_index = record.get("step_index", idx)
                source = str(record.get("source", "UNKNOWN"))
                msg_type = str(record.get("type", "UNKNOWN"))
                created_at = record.get("created_at") or file_dt_str
                content = sanitize_text_content(record.get("content", ""))
                thinking = record.get("thinking")
                has_thinking = 1 if (thinking and str(thinking).strip()) else 0
                tool_names = extract_tool_names(record.get("tool_calls"))

                if not earliest_time:
                    earliest_time = created_at
                latest_time = created_at

                if not derived_title and msg_type == "USER_INPUT" and content:
                    first_line = content.splitlines()[0].strip()
                    # Strip markdown heading marks
                    first_line = first_line.lstrip("#").strip()
                    derived_title = first_line[:100]

                messages.append({
                    "conversation_id": conversation_id,
                    "step_index": step_index,
                    "created_at": created_at,
                    "source": source,
                    "type": msg_type,
                    "content": content,
                    "tool_names": tool_names,
                    "has_thinking": has_thinking,
                })
    except Exception as e:
        logger.warning("Error reading %s: %s", transcript_path, e)

    if not derived_title:
        derived_title = f"Conversation {conversation_id[:8]}"

    conversation_meta = {
        "conversation_id": conversation_id,
        "title": derived_title,
        "created_at": earliest_time or file_dt_str,
        "updated_at": latest_time or file_dt_str,
        "total_steps": len(messages),
        "file_path": str(transcript_path),
        "last_indexed_mtime": file_mtime,
    }

    return conversation_meta, messages


def find_transcript_files(brain_dir: Path) -> List[Tuple[str, Path]]:
    """Locate all transcript.jsonl files under the brain directory."""
    results: List[Tuple[str, Path]] = []
    if not brain_dir.exists() or not brain_dir.is_dir():
        return results

    try:
        for entry in brain_dir.iterdir():
            if entry.is_dir():
                conv_id = entry.name
                log_dir = entry / ".system_generated" / "logs"
                transcript_path = log_dir / "transcript.jsonl"
                if transcript_path.is_file():
                    results.append((conv_id, transcript_path))
                else:
                    # Fallback to transcript_full.jsonl if standard transcript is absent
                    full_path = log_dir / "transcript_full.jsonl"
                    if full_path.is_file():
                        results.append((conv_id, full_path))
    except Exception as e:
        logger.warning("Error scanning brain directory %s: %s", brain_dir, e)

    return results


def sync_index(
    brain_dir: Optional[Path | str] = None,
    db_path: Optional[Path | str] = None,
    force_rescan: bool = False,
) -> Dict[str, Any]:
    """Incrementally synchronize transcript files into the SQLite database."""
    start_time = time.time()
    target_brain = Path(brain_dir) if brain_dir else get_brain_dir()
    conn = get_db_connection(db_path)
    init_db(conn)

    transcript_items = find_transcript_files(target_brain)
    stats = {
        "discovered_transcripts": len(transcript_items),
        "indexed_new": 0,
        "updated": 0,
        "skipped": 0,
        "total_messages": 0,
        "duration_seconds": 0.0,
        "brain_dir": str(target_brain),
    }

    # Query currently indexed conversation modification times
    existing_mtimes: Dict[str, float] = {}
    if not force_rescan:
        rows = conn.execute("SELECT conversation_id, last_indexed_mtime FROM conversations").fetchall()
        for r in rows:
            existing_mtimes[r["conversation_id"]] = r["last_indexed_mtime"] or 0.0

    for conv_id, transcript_file in transcript_items:
        try:
            current_mtime = transcript_file.stat().st_mtime
        except OSError:
            continue

        prev_mtime = existing_mtimes.get(conv_id)
        if not force_rescan and prev_mtime is not None and prev_mtime >= current_mtime:
            stats["skipped"] += 1
            continue

        meta, messages = parse_transcript_file(transcript_file, conv_id)
        is_update = prev_mtime is not None

        with conn:
            # Delete existing records if updating
            if is_update:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
                conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conv_id,))

            # Insert conversation metadata
            conn.execute("""
                INSERT INTO conversations (
                    conversation_id, title, created_at, updated_at, total_steps, file_path, last_indexed_mtime
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                meta["conversation_id"],
                meta["title"],
                meta["created_at"],
                meta["updated_at"],
                meta["total_steps"],
                meta["file_path"],
                meta["last_indexed_mtime"],
            ))

            # Batch insert messages
            if messages:
                conn.executemany("""
                    INSERT INTO messages (
                        conversation_id, step_index, created_at, source, type, content, tool_names, has_thinking
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        m["conversation_id"],
                        m["step_index"],
                        m["created_at"],
                        m["source"],
                        m["type"],
                        m["content"],
                        m["tool_names"],
                        m["has_thinking"],
                    )
                    for m in messages
                ])

        if is_update:
            stats["updated"] += 1
        else:
            stats["indexed_new"] += 1
        stats["total_messages"] += len(messages)

    conn.close()
    stats["duration_seconds"] = round(time.time() - start_time, 3)
    return stats
