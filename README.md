# search-antigravity

**A Dedicated FastMCP Full-Text Search Engine for Google Antigravity Session Transcripts**

`search-antigravity` is a lightweight Model Context Protocol (MCP) server that indexes historical Google Antigravity session transcripts from `~/.gemini/antigravity/brain/<conversation-id>/.system_generated/logs/transcript.jsonl` into an embedded SQLite database with full-text search (FTS5) and BM25 relevance ranking.

It enables agents and human users to execute surgical, token-efficient queries across historical discussions, recovering context, decisions, and code snippets in ~50–100 tokens.

---

## Features

* **Incremental MTime-Based Indexing:** Scans session folders and only re-indexes modified or newly added transcripts.
* **SQLite + FTS5 BM25 Search Engine:** Fast full-text search with keyword highlighting, phrase matching (`"exact phrase"`), prefix queries (`term*`), and boolean operators (`AND`, `OR`, `NOT`).
* **Noise Mitigation:** Strips and truncates oversized binary/tool outputs to keep the search index compact and high-signal.
* **Dual MCP SDK Compatibility:** Seamlessly supports both MCP Python SDK 2.x (`MCPServer`) and 1.x (`FastMCP`).
* **5 Core MCP Tools:**
  1. `search_antigravity_conversations`: Search across conversations with BM25 snippet highlights.
  2. `get_antigravity_step`: Inspect dialogue turns before/after any target step index.
  3. `list_antigravity_conversations`: Browse recent conversations and activity ranges.
  4. `get_antigravity_stats`: Summary metrics, message counts by type, date span, and database size.
  5. `sync_antigravity_index`: Trigger on-demand incremental indexing of local sessions.

---

## Project Structure

```text
search-antigravity/
├── pyproject.toml              # Build & dependency metadata
├── requirements.txt            # Core dependencies (mcp, pydantic, pytest)
├── run_sync.py                 # Standalone manual indexing CLI
├── README.md                   # Documentation
├── src/
│   ├── __init__.py
│   ├── db.py                   # SQLite connection, schema, FTS5 triggers, queries
│   ├── indexer.py              # Incremental scanner & JSONL transcript parser
│   └── server.py               # FastMCP server & MCP tool registrations
└── tests/
    └── test_search.py          # Pytest suite with synthetic transcripts
```

---

## Installation & Setup

### 1. Create Virtual Environment and Install Dependencies

```bash
cd /path/to/search-antigravity
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Initial Indexing (CLI)

```bash
python run_sync.py
```

Options:
* `--brain-dir <PATH>`: Custom brain directory (default: `~/.gemini/antigravity/brain`).
* `--db-path <PATH>`: Custom database path (default: `./conversations.db`).
* `--force`: Force full re-indexing of all sessions.

---

## Automated Background Synchronization

`search-antigravity` synchronizes transcripts automatically without requiring manual commands:
* **Periodic Background Daemon:** A background thread periodically scans and indexes new turns every **10 minutes** (configurable via `ANTIGRAVITY_SYNC_INTERVAL_SECONDS`).
* **Just-In-Time Refresh:** Running `search_antigravity_conversations` executes a quick 15ms incremental refresh if 30 seconds have passed since the last sync.
* **Agent-Triggered:** Agents can invoke `sync_antigravity_index` on demand at any point.

---

## Antigravity IDE Configuration

Add `search-antigravity` to your global MCP configuration in `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "search-antigravity": {
      "command": "/path/to/search-antigravity/.venv/bin/python",
      "args": [
        "-m",
        "src.server"
      ],
      "cwd": "/path/to/search-antigravity"
    }
  }
}
```


---

## MCP Tools Reference

### `search_antigravity_conversations`
Searches messages across all indexed Antigravity conversations.
* **Parameters:**
  * `query` *(string, required)*: Keywords, exact phrases (`"search term"`), or booleans (`python AND sqlite`).
  * `conversation_id` *(string, optional)*: Restrict search to a specific session.
  * `type_filter` *(string, optional)*: Filter by type (`USER_INPUT`, `PLANNER_RESPONSE`, etc.).
  * `limit` *(int, default: 10)*: Number of matches (1 to 50).

### `get_antigravity_step`
Retrieves dialogue turns surrounding a specific step.
* **Parameters:**
  * `conversation_id` *(string, required)*: ID of the conversation.
  * `step_index` *(int, required)*: Step index to center around.
  * `context_window` *(int, default: 2)*: Preceding and succeeding turns to include.

### `list_antigravity_conversations`
Lists recent conversations sorted by latest activity.
* **Parameters:**
  * `limit` *(int, default: 20)*
  * `offset` *(int, default: 0)*

### `get_antigravity_stats`
Returns total indexed conversations, message breakdown by type, date ranges, and database disk usage.

### `sync_antigravity_index`
Triggers an incremental scan of `~/.gemini/antigravity/brain/` and returns indexing metrics.

---

## Testing
 
```bash
.venv/bin/pytest tests/ -v
```

---

## Privacy, Anonymization & Public Showcase

* **Local-Only & Private:** The SQLite database (`conversations.db*`) stores parsed session logs strictly on your local filesystem and is ignored by `.gitignore`. Nothing is sent to external servers.
* **Showcase / Demo Dataset Generator:** If you want to demonstrate or showcase this project publicly without exposing personal transcripts or proprietary code, you can generate a clean, synthetic dataset:
  ```bash
  # 1. Generate realistic synthetic developer discussions
  python scripts/generate_sample_data.py

  # 2. Index the demo dataset into a standalone demo database
  python run_sync.py --brain-dir data/sample_brain --db-path data/demo.db

  # 3. Launch the MCP server using the demo database
  ANTIGRAVITY_DB_PATH=data/demo.db python -m src.server
  ```
  *(Or inspect tools interactively via MCP Inspector)*:
  ```bash
  ANTIGRAVITY_DB_PATH=data/demo.db npx @modelcontextprotocol/inspector .venv/bin/python -m src.server
  ```


