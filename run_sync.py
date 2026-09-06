#!/usr/bin/env python3
"""Standalone CLI script to manually synchronize Antigravity transcripts into SQLite FTS5 index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.db import get_database_stats, get_db_connection, get_db_path
from src.indexer import get_brain_dir, sync_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize local Antigravity session transcripts into SQLite FTS5 database."
    )
    parser.add_argument(
        "--brain-dir",
        type=str,
        default=None,
        help="Path to Antigravity brain directory (default: ~/.gemini/antigravity/brain)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to conversations.db (default: conversations.db in project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all transcripts even if modification timestamp hasn't changed.",
    )

    args = parser.parse_args()

    brain_dir = Path(args.brain_dir) if args.brain_dir else get_brain_dir()
    db_path = Path(args.db_path) if args.db_path else get_db_path()

    print(f"🚀 Starting Antigravity Transcript Indexing...")
    print(f"   Brain Directory: {brain_dir}")
    print(f"   Database Path:   {db_path}")
    print(f"   Force Rescan:    {args.force}")
    print()

    if not brain_dir.exists():
        print(f"⚠️  Brain directory does not exist: {brain_dir}")
        print("   If your transcripts are in another location, specify with --brain-dir or ANTIGRAVITY_BRAIN_DIR.")
        return 1

    stats = sync_index(brain_dir=brain_dir, db_path=db_path, force_rescan=args.force)

    print("✅ Indexing Complete!")
    print(f"   - Discovered Transcripts: {stats['discovered_transcripts']}")
    print(f"   - Newly Indexed:          {stats['indexed_new']}")
    print(f"   - Updated Conversations:  {stats['updated']}")
    print(f"   - Skipped (Unchanged):    {stats['skipped']}")
    print(f"   - Total Messages Processed: {stats['total_messages']}")
    print(f"   - Duration:               {stats['duration_seconds']}s")
    print()

    conn = get_db_connection(db_path)
    try:
        db_stats = get_database_stats(conn, db_path)
        print("📊 Current Index Statistics:")
        print(f"   - Total Conversations:    {db_stats['total_conversations']:,}")
        print(f"   - Total Messages:         {db_stats['total_messages']:,}")
        print(f"   - Database Size:          {db_stats['db_size_mb']} MB")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
