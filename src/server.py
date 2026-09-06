"""FastMCP Server exposing search and context retrieval tools for Antigravity transcripts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Compatibility between MCP SDK 2.x (MCPServer) and 1.x (FastMCP)
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from src.db import (
    get_database_stats,
    get_db_connection,
    get_db_path,
    get_step_context,
    init_db,
    list_recent_conversations,
    search_messages,
)
from src.indexer import get_brain_dir, sync_index
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("search-antigravity")

DEFAULT_SYNC_INTERVAL_SECONDS = int(os.environ.get("ANTIGRAVITY_SYNC_INTERVAL_SECONDS", "600"))  # 10 minutes
_last_sync_time: float = 0.0
_sync_lock = threading.Lock()


def trigger_background_sync(force: bool = False) -> None:
    """Run incremental sync if cooldown interval has elapsed or forced."""
    global _last_sync_time
    with _sync_lock:
        now = time.time()
        # Cooldown of 30 seconds between auto-checks
        if not force and (now - _last_sync_time) < 30:
            return
        try:
            sync_index()
            _last_sync_time = time.time()
        except Exception as e:
            logger.warning("Background sync error: %s", e)


def _background_sync_loop(interval_seconds: int) -> None:
    """Daemon thread loop that periodically syncs the index."""
    logger.info("Background auto-sync thread started (interval: %ds)", interval_seconds)
    while True:
        time.sleep(interval_seconds)
        trigger_background_sync(force=True)


def start_background_syncer(interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS) -> None:
    """Start background sync daemon thread if interval > 0."""
    if interval_seconds > 0:
        thread = threading.Thread(target=_background_sync_loop, args=(interval_seconds,), daemon=True)
        thread.start()


mcp = FastMCP("search-antigravity")



@mcp.tool()
def search_antigravity_conversations(
    query: str,
    conversation_id: Optional[str] = None,
    type_filter: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Primary search engine for Antigravity conversations using FTS5 BM25 relevance ranking.

    Args:
        query: Search keywords, phrases (e.g. '"error in model"'), or boolean operators (AND, OR, NOT, wildcard*).
        conversation_id: Optional ID to restrict search within a single conversation session.
        type_filter: Optional filter by step type: 'USER_INPUT' (prompts), 'PLANNER_RESPONSE' (model actions), etc.
        limit: Maximum results to return (default 10, max 50).

    Returns:
        Ranked list of conversation matches with snippets, step indices, timestamps, and conversation IDs.
    """
    db_path = get_db_path()
    if not db_path.exists():
        logger.info("Database not found on first search. Running initial sync...")
        sync_index()
    else:
        # Trigger quick incremental refresh if cooldown has elapsed
        trigger_background_sync(force=False)


    conn = get_db_connection()
    try:
        results = search_messages(
            conn=conn,
            query=query,
            conversation_id=conversation_id,
            type_filter=type_filter,
            limit=limit,
        )

        if not results:
            msg = f"No results found for query: '{query}'"
            if conversation_id:
                msg += f" in conversation `{conversation_id}`"
            if type_filter:
                msg += f" with type '{type_filter}'"
            return msg + "."

        lines = [f"### 🔍 Search Results for: '{query}' ({len(results)} matches)\n"]
        for idx, r in enumerate(results, 1):
            title = r.get("title") or "Untitled Conversation"
            conv_id = r["conversation_id"]
            step = r["step_index"]
            step_type = r["type"]
            created = r["created_at"]
            snippet = r["snippet"].replace("\n", " ")
            tools = r.get("tool_names")

            line = f"{idx}. **{title}** (`{conv_id}`) - Step #{step} `[{step_type}]`\n"
            line += f"   - **Timestamp:** {created}\n"
            if tools:
                line += f"   - **Tools Used:** `{tools}`\n"
            line += f"   - **Snippet:** {snippet}\n"
            lines.append(line)

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def get_antigravity_step(
    conversation_id: str,
    step_index: int,
    context_window: int = 2,
) -> str:
    """Retrieves full dialogue context surrounding a specific step discovered in search.

    Args:
        conversation_id: The conversation ID containing the step.
        step_index: The specific step index to inspect.
        context_window: Number of steps before and after to include for dialogue flow (default 2).

    Returns:
        Formatted dialogue exchange for that specific window.
    """
    conn = get_db_connection()
    try:
        steps = get_step_context(
            conn=conn,
            conversation_id=conversation_id,
            step_index=step_index,
            context_window=context_window,
        )

        if not steps:
            return f"No steps found for conversation `{conversation_id}` near step #{step_index}."

        title = steps[0].get("title") or conversation_id
        min_step = steps[0]["step_index"]
        max_step = steps[-1]["step_index"]

        lines = [
            f"### 💬 Conversation Context: `{conversation_id}`",
            f"**Title:** {title}",
            f"**Window:** Steps #{min_step} to #{max_step} (Target: #{step_index})\n",
        ]

        for s in steps:
            is_target = " 🎯 [TARGET STEP]" if s["step_index"] == step_index else ""
            header = f"#### Step #{s['step_index']}: {s['source']} ({s['type']}){is_target}"
            lines.append(header)
            lines.append(f"*Timestamp: {s['created_at']}*")
            if s.get("tool_names"):
                lines.append(f"*Tools: `{s['tool_names']}`*")
            lines.append("")
            lines.append(s["content"] or "*(empty content)*")
            lines.append("\n---\n")

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def list_antigravity_conversations(
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Lists recent Antigravity conversation sessions sorted by latest activity.

    Args:
        limit: Maximum conversations to list (default 20, max 100).
        offset: Pagination offset (default 0).

    Returns:
        List of conversations with IDs, titles, step counts, and active date ranges.
    """
    db_path = get_db_path()
    if not db_path.exists():
        sync_index()

    conn = get_db_connection()
    try:
        conversations = list_recent_conversations(conn, limit=limit, offset=offset)
        if not conversations:
            return "No conversations indexed yet. Run `sync_antigravity_index` to index sessions."

        lines = [f"### 📂 Recent Antigravity Conversations ({offset + 1} - {offset + len(conversations)})\n"]
        for idx, c in enumerate(conversations, offset + 1):
            title = c["title"] or "Untitled"
            conv_id = c["conversation_id"]
            steps = c["total_steps"]
            created = c["created_at"]
            updated = c["updated_at"]

            lines.append(
                f"{idx}. **{title}**\n"
                f"   - **ID:** `{conv_id}`\n"
                f"   - **Total Steps:** {steps}\n"
                f"   - **Timeline:** {created} → {updated}\n"
            )

        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def get_antigravity_stats() -> str:
    """Returns health, count, and disk storage metrics for the indexed transcripts.

    Returns:
        Summary metrics including total sessions, messages, breakdown by type, and database size.
    """
    db_path = get_db_path()
    if not db_path.exists():
        sync_index()

    conn = get_db_connection()
    try:
        stats = get_database_stats(conn, db_path)
        types_breakdown = stats.get("messages_by_type", {})
        breakdown_str = "\n".join([f"    - **{k}:** {v:,}" for k, v in types_breakdown.items()]) or "    *(none)*"

        return (
            f"### 📊 Antigravity Index Statistics\n\n"
            f"- **Total Conversations:** {stats['total_conversations']:,}\n"
            f"- **Total Messages:** {stats['total_messages']:,}\n"
            f"  - **Breakdown by Type:**\n{breakdown_str}\n"
            f"- **Date Range:** {stats['earliest_date'] or 'N/A'} → {stats['latest_date'] or 'N/A'}\n"
            f"- **Database Size:** {stats['db_size_mb']:.2f} MB ({stats['db_size_bytes']:,} bytes)\n"
            f"- **Database Path:** `{stats['db_path']}`\n"
            f"- **Brain Source Directory:** `{get_brain_dir()}`\n"
        )
    finally:
        conn.close()


@mcp.tool()
def sync_antigravity_index() -> str:
    """Triggers an on-demand incremental scan of the Antigravity brain transcripts.

    Returns:
        Summary of newly indexed, updated, and skipped conversations.
    """
    res = sync_index()
    return (
        f"### 🔄 Index Sync Completed in {res['duration_seconds']}s\n\n"
        f"- **Transcripts Discovered:** {res['discovered_transcripts']}\n"
        f"- **New Conversations Indexed:** {res['indexed_new']}\n"
        f"- **Updated Conversations:** {res['updated']}\n"
        f"- **Unchanged / Skipped:** {res['skipped']}\n"
        f"- **Total Messages Processed:** {res['total_messages']}\n"
        f"- **Brain Directory:** `{res['brain_dir']}`\n"
    )


def main() -> None:
    """Server entrypoint running over stdio transport."""
    # Ensure database schema is ready and run non-blocking initial scan if DB doesn't exist
    db_path = get_db_path()
    if not db_path.exists():
        logger.info("Initializing database and performing initial sync...")
        try:
            sync_index()
        except Exception as e:
            logger.warning("Initial index sync failed: %s", e)

    # Start periodic background synchronization (default: 600s / 10 minutes)
    start_background_syncer()

    mcp.run(transport="stdio")



if __name__ == "__main__":
    main()
