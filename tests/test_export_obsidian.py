"""
Verify Obsidian mirror export is idempotent and mirrors tombstones.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_export_obsidian.py
"""
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("./.data/chroma_export_obsidian_test")
OUT_PATH = Path("./.data/obsidian_export_test")
SCRIPT_PATH = Path("scripts/export_obsidian.py")

for path in (DB_PATH, OUT_PATH):
    if path.exists():
        shutil.rmtree(path)
DB_PATH.mkdir(parents=True)

spec = importlib.util.spec_from_file_location("export_obsidian", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["export_obsidian"] = module
spec.loader.exec_module(module)

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    conn.execute(
        """
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tombstoned INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO messages(message_id, role, text, content_hash, created_at, tombstoned)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "user", "Active mirror message.", "hash-active", "2026-01-01T00:00:00+00:00", 0),
            (2, "assistant", "Forgotten mirror message.", "hash-forgotten", "2026-01-02T00:00:00+00:00", 1),
            (3, "user", "Second active mirror message.", "hash-active-two", "2026-01-03T00:00:00+00:00", 0),
        ],
    )

structured = [
    {
        "id": "sm_active",
        "type": "decision",
        "summary": "Active structured item.",
        "source_text": "Active mirror message.",
        "tags": ["mirror"],
        "importance": 0.8,
        "message_id": 1,
        "role": "user",
    },
    {
        "id": "sm_forgotten",
        "type": "preference",
        "summary": "Forgotten structured item.",
        "source_text": "Forgotten mirror message.",
        "tags": ["mirror"],
        "importance": 0.7,
        "message_id": 2,
        "role": "assistant",
    },
]
(DB_PATH / "structured_memory.jsonl").write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in structured) + "\n",
    encoding="utf-8",
)

first = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2)
assert first["messages"] == 3
assert first["structured"] == 2

active_msg = OUT_PATH / "active/messages/msg-000001.md"
forgotten_msg = OUT_PATH / "forgotten/messages/msg-000002.md"
second_active_msg = OUT_PATH / "active/messages/msg-000003.md"
active_structured = OUT_PATH / "active/structured/sm_active.md"
forgotten_structured = OUT_PATH / "forgotten/structured/sm_forgotten.md"
timeline = OUT_PATH / "maps/timeline-0001-0002.md"
turns = OUT_PATH / "maps/turns.md"

assert active_msg.exists()
assert forgotten_msg.exists()
assert second_active_msg.exists()
assert active_structured.exists()
assert forgotten_structured.exists()
active_text = active_msg.read_text(encoding="utf-8")
forgotten_text = forgotten_msg.read_text(encoding="utf-8")
second_active_text = second_active_msg.read_text(encoding="utf-8")
assert "[[sm_active]]" in active_text
assert "Next: [[msg-000002]]" in active_text
assert "Previous: [[msg-000001]]" in forgotten_text
assert "Next: [[msg-000003]]" in forgotten_text
assert "Previous: [[msg-000002]]" in second_active_text
assert "[[msg-000001]]" in active_structured.read_text(encoding="utf-8")
assert "status: \"forgotten\"" in forgotten_text
assert timeline.exists()
timeline_text = timeline.read_text(encoding="utf-8")
assert "[[msg-000001]]" in timeline_text
assert "[[msg-000003]]" in timeline_text
assert "[[msg-000002]]" not in timeline_text
turns_text = turns.read_text(encoding="utf-8")
assert "Rule: one user message plus contiguous following assistant messages" in turns_text
assert "[[msg-000001]]" in turns_text
assert "[[msg-000003]]" in turns_text

mtimes = {
    path: path.stat().st_mtime_ns
    for path in (
        active_msg,
        forgotten_msg,
        second_active_msg,
        active_structured,
        forgotten_structured,
        timeline,
        turns,
        OUT_PATH / "index.md",
    )
}
second = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2)
assert second["written"] == 0
assert second["removed"] == 0
assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    conn.execute("UPDATE messages SET tombstoned = 1 WHERE message_id = 1")

third = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2)
assert third["removed"] == 2
assert not active_msg.exists()
assert not active_structured.exists()
assert (OUT_PATH / "forgotten/messages/msg-000001.md").exists()
assert (OUT_PATH / "forgotten/structured/sm_active.md").exists()

print("Obsidian export test passed.")
