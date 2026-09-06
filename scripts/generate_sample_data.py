#!/usr/bin/env python3
"""Generate anonymized, realistic synthetic Antigravity transcripts for demos and showcases."""

from __future__ import annotations

import json
from pathlib import Path

SAMPLE_CONVERSATIONS = [
    {
        "id": "demo-001-microservices-bm25",
        "steps": [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-09-01T10:15:00Z",
                "content": "# Architecture Design: BM25 Full-Text Search Engine\nWe need to build a lightweight FastMCP service in Python that indexes our session logs using SQLite FTS5 with BM25 ranking. How should we structure the database schema and triggers?",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-09-01T10:15:30Z",
                "content": "To achieve high search throughput and zero index bloat, I recommend an SQLite database in WAL mode with an FTS5 virtual table synchronized via automatic triggers.\n\nHere is the proposed schema:\n```sql\nCREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id TEXT, step_index INTEGER, content TEXT, tool_names TEXT);\nCREATE VIRTUAL TABLE messages_fts USING fts5(content, tool_names, tokenize = 'porter unicode61');\n```",
                "thinking": "The user requires sub-second queries with minimal disk footprint. SQLite FTS5 BM25 is ideal.",
                "tool_calls": [{"name": "write_to_file", "args": {"TargetFile": "src/db.py"}}],
            },
            {
                "step_index": 2,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-09-01T10:16:00Z",
                "content": "Make sure we also support incremental indexing by checking file modification timestamps (`mtime`).",
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-09-01T10:16:45Z",
                "content": "Implemented incremental parser in `src/indexer.py`. It inspects file `mtime` against `last_indexed_mtime`, skipping unchanged sessions in ~15ms.",
                "tool_calls": [{"name": "run_command", "args": {"CommandLine": "pytest tests/"}}],
            },
        ],
    },
    {
        "id": "demo-002-asyncio-memory-leak",
        "steps": [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-09-02T14:20:00Z",
                "content": "We noticed memory usage steadily climbing under heavy load in our Python asyncio worker. Can you profile the event loop and find uncollected tasks?",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-09-02T14:21:10Z",
                "content": "I analyzed the heap dump with tracemalloc. Background tasks created with `asyncio.create_task()` were not retaining strong references, leading to lingering task objects and unclosed sockets.",
                "thinking": "Check strong reference retention in global task set.",
                "tool_calls": [{"name": "run_command", "args": {"CommandLine": "python3 -m pytest tests/test_leak.py"}}],
            },
            {
                "step_index": 2,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-09-02T14:25:00Z",
                "content": "Nice catch! Memory is flat at 45MB now.",
            },
        ],
    },
    {
        "id": "demo-003-k8s-autoscaling-helm",
        "steps": [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-09-03T09:00:00Z",
                "content": "Set up a Kubernetes Horizontal Pod Autoscaler (HPA) using custom Prometheus metrics for our API deployment.",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-09-03T09:01:20Z",
                "content": "Created Helm values for the `prometheus-adapter` and defined HPA target of 80% CPU and 500 requests/second threshold.",
                "tool_calls": [{"name": "write_to_file", "args": {"TargetFile": "helm/values.yaml"}}],
            },
        ],
    },
]


def generate_sample_brain(output_dir: Path) -> None:
    """Generate sample conversations inside output_dir imitating ~/.gemini/antigravity/brain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for conv in SAMPLE_CONVERSATIONS:
        conv_dir = output_dir / conv["id"] / ".system_generated" / "logs"
        conv_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = conv_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for step in conv["steps"]:
                f.write(json.dumps(step) + "\n")
    print(f"✅ Generated {len(SAMPLE_CONVERSATIONS)} synthetic conversations in: {output_dir}")


def main() -> None:
    default_dir = Path(__file__).resolve().parent.parent / "data" / "sample_brain"
    generate_sample_brain(default_dir)
    print("\nTo index this sample dataset for a demo or public showcase, run:")
    print(f"  python run_sync.py --brain-dir data/sample_brain --db-path demo.db")
    print("  search-antigravity (with ANTIGRAVITY_DB_PATH=demo.db)")


if __name__ == "__main__":
    main()
