"""
Verify Obsidian mirror export is idempotent and mirrors tombstones.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_export_obsidian.py
"""
import importlib.util
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("./.data/chroma_export_obsidian_test")
OUT_PATH = Path("./.data/obsidian_export_test")
CONFIG_PATH = DB_PATH / "ragmemory.local.ini"
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
            compact_text TEXT,
            compact_status TEXT,
            tombstoned INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO messages(
            message_id, role, text, content_hash, created_at,
            compact_text, compact_status, tombstoned
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "user",
                "Active mirror message raw should stay in SQLite.",
                "hash-active",
                "2026-01-01T00:00:00+00:00",
                "Compact active mirror message. See evidence[python:abc123abc123].",
                "ok",
                0,
            ),
            (
                2,
                "assistant",
                "Forgotten mirror message.",
                "hash-forgotten",
                "2026-01-02T00:00:00+00:00",
                None,
                None,
                1,
            ),
            (
                3,
                "user",
                "Second active mirror message with example [[files/unwanted-test-hub]].",
                "hash-active-two",
                "2026-01-03T00:00:00+00:00",
                None,
                None,
                0,
            ),
        ],
    )

structured = [
    {
        "id": "sm_active",
        "type": "decision",
        "summary": "Active structured item.",
        "source_text": "Active mirror message.",
        "tags": ["mirror", "C++", "C  ++"],
        "importance": 0.8,
        "message_id": 1,
        "role": "user",
    },
    {
        "id": "sm_code",
        "type": "code_reference",
        "summary": "Memory code reference.",
        "source_text": "Review src/ragmemory/memory.py before changing retrieval.",
        "tags": ["python", "RagMemory"],
        "importance": 0.9,
        "message_id": 1,
        "role": "user",
        "content_hash": "abc123abc123",
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
    {
        "id": "sm_repeat",
        "type": "decision",
        "summary": "Repeated mirror tag.",
        "source_text": "Second active mirror message.",
        "tags": ["mirror", "singleton"],
        "importance": 0.7,
        "message_id": 3,
        "role": "user",
    },
    {
        "id": "sm_deny",
        "type": "decision",
        "summary": "Denylisted tags do not become hubs.",
        "source_text": "Generic artifact tag example.",
        "tags": ["code_reference", "python"],
        "importance": 0.7,
        "message_id": 3,
        "role": "user",
    },
    {
        "id": "sm_profile",
        "type": "preference",
        "summary": "Active profile preference.",
        "source_text": "User prefers focused graph hubs.",
        "tags": ["profile", "personal"],
        "importance": 0.7,
        "message_id": 3,
        "role": "user",
    },
]
(DB_PATH / "structured_memory.jsonl").write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in structured) + "\n",
    encoding="utf-8",
)
CONFIG_PATH.write_text(
    """
[obsidian.topics]
min_count = 2
allowlist = ragmemory, mirror, c
denylist = code_reference, config, decision, constraint, preference,
           open_question, chart, table, text, profile,
           python, powershell, javascript, typescript, bash,
           ini, yaml, json, markdown, sql, html, css

[obsidian.files]
enable = true
""".strip()
    + "\n",
    encoding="utf-8",
)

first = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2, config_path=CONFIG_PATH)
assert first["messages"] == 3
assert first["structured"] == 6

active_msg = OUT_PATH / "active/messages/msg-000001.md"
forgotten_msg = OUT_PATH / "forgotten/messages/msg-000002.md"
second_active_msg = OUT_PATH / "active/messages/msg-000003.md"
active_structured = OUT_PATH / "active/structured/sm_active.md"
code_structured = OUT_PATH / "active/structured/sm_code.md"
forgotten_structured = OUT_PATH / "forgotten/structured/sm_forgotten.md"
timeline = OUT_PATH / "maps/timeline-0001-0002.md"
turns = OUT_PATH / "maps/turns.md"
topic_mirror = OUT_PATH / "topics/mirror.md"
topic_ragmemory = OUT_PATH / "topics/ragmemory.md"
topic_c = OUT_PATH / "topics/c.md"
topic_collision = OUT_PATH / f"topics/c-{module.stable_suffix('c++')}.md"
file_hub = OUT_PATH / "files/src-ragmemory-memory-py.md"
profile_hub = OUT_PATH / "profile/user.md"
topic_singleton = OUT_PATH / "topics/singleton.md"
topic_code_reference = OUT_PATH / "topics/code-reference.md"
topic_python = OUT_PATH / "topics/python.md"
topic_profile = OUT_PATH / "topics/profile.md"

assert active_msg.exists()
assert forgotten_msg.exists()
assert second_active_msg.exists()
assert active_structured.exists()
assert code_structured.exists()
assert forgotten_structured.exists()
assert topic_mirror.exists()
assert topic_ragmemory.exists()
assert topic_c.exists()
assert topic_collision.exists()
assert file_hub.exists()
assert profile_hub.exists()
assert not topic_singleton.exists()
assert not topic_code_reference.exists()
assert not topic_python.exists()
assert not topic_profile.exists()
active_text = active_msg.read_text(encoding="utf-8")
forgotten_text = forgotten_msg.read_text(encoding="utf-8")
second_active_text = second_active_msg.read_text(encoding="utf-8")
assert "[[sm_active]]" in active_text
assert "[[sm_code]]" in active_text
assert "[[msg-" not in active_text
assert "Text source: `compact_text`" in active_text
assert "Compact active mirror message." in active_text
assert "## Evidence References" in active_text
assert "`evidence[python:abc123abc123]` -> [[sm_code]]" in active_text
assert "raw should stay in SQLite" not in active_text
assert "next_message: \"msg-000002\"" in active_text
assert "previous_message: \"msg-000001\"" in forgotten_text
assert "next_message: \"msg-000003\"" in forgotten_text
assert "[[msg-" not in second_active_text
assert "[[files/unwanted-test-hub]]" not in second_active_text
assert "&#91;&#91;files/unwanted-test-hub&#93;&#93;" in second_active_text
raw_only_text = module.message_markdown(
    module.MessageRow(99, "user", "raw only", "hash-raw", None, False),
    [],
)
assert "cssclasses: [\"memory-message\", \"memory-unlinked\"]" in raw_only_text
assert "[[msg-000001]]" in active_structured.read_text(encoding="utf-8")
code_text = code_structured.read_text(encoding="utf-8")
assert "content_hash: \"abc123abc123\"" in code_text
assert "evidence_ref: \"evidence[python:abc123abc123]\"" in code_text
assert "- Evidence ref: `evidence[python:abc123abc123]`" in code_text
assert "[[topics/ragmemory]]" in code_text
assert "[[files/src-ragmemory-memory-py]]" in code_text
assert "[[topics/mirror]]" in active_structured.read_text(encoding="utf-8")
assert "[[topics/c]]" in active_structured.read_text(encoding="utf-8")
assert f"[[topics/c-{module.stable_suffix('c++')}]]" in active_structured.read_text(encoding="utf-8")
assert "[[topics/singleton]]" not in (OUT_PATH / "active/structured/sm_repeat.md").read_text(encoding="utf-8")
assert "[[topics/code-reference]]" not in (OUT_PATH / "active/structured/sm_deny.md").read_text(encoding="utf-8")
assert "[[topics/python]]" not in (OUT_PATH / "active/structured/sm_deny.md").read_text(encoding="utf-8")
assert "[[topics/profile]]" not in (OUT_PATH / "active/structured/sm_profile.md").read_text(encoding="utf-8")
assert "[[profile/user]]" in (OUT_PATH / "active/structured/sm_profile.md").read_text(encoding="utf-8")
disabled_file_policy = module.FileHubPolicy(enabled=False)
disabled_registry = module.build_hub_registry(
    module.load_structured(DB_PATH),
    module.load_topic_policy(CONFIG_PATH),
    module.topic_counts(module.load_structured(DB_PATH)),
    disabled_file_policy,
)
assert all(hub.hub_type != "file" for hub in disabled_registry.values())
forgotten_structured_text = forgotten_structured.read_text(encoding="utf-8")
assert "[[profile/user]]" in forgotten_structured_text
assert "tags:" not in forgotten_structured_text.split("---", 2)[1]
assert "status: \"forgotten\"" in forgotten_text
assert timeline.exists()
timeline_text = timeline.read_text(encoding="utf-8")
assert "cssclasses: [\"navigation\"]" in timeline_text
assert "[[msg-000001]]" in timeline_text
assert "Compact active mirror message." in timeline_text
assert "[[msg-000003]]" in timeline_text
assert "[[msg-000002]]" not in timeline_text
turns_text = turns.read_text(encoding="utf-8")
assert "cssclasses: [\"navigation\"]" in turns_text
assert "Rule: one user message plus contiguous following assistant messages" in turns_text
assert "[[msg-000001]]" in turns_text
assert "[[msg-000003]]" in turns_text
assert "cssclasses: [\"navigation\"]" in (OUT_PATH / "index.md").read_text(encoding="utf-8")

note_stems = set()
for note in OUT_PATH.rglob("*.md"):
    relative_stem = note.relative_to(OUT_PATH).with_suffix("").as_posix()
    note_stems.add(relative_stem)
    note_stems.add(note.stem)

all_edges = []
for note in OUT_PATH.rglob("*.md"):
    text = note.read_text(encoding="utf-8")
    frontmatter_text = text.split("---", 2)[1] if text.startswith("---") else ""
    body = text.split("---", 2)[2] if text.startswith("---") else text
    source = note.relative_to(OUT_PATH).with_suffix("").as_posix()
    is_navigation = "cssclasses: [\"navigation\"]" in frontmatter_text
    for target in re.findall(r"\[\[([^|\]#]+)", body):
        all_edges.append((source, target, is_navigation))
        assert target in note_stems, f"Unresolved wikilink target: {target}"

chronology_edges = [
    edge for edge in all_edges
    if not edge[2]
    and Path(edge[0]).name.startswith("msg-")
    and Path(edge[1]).name.startswith("msg-")
]
message_to_structured_edges = [
    edge for edge in all_edges
    if not edge[2]
    and Path(edge[0]).name.startswith("msg-")
    and edge[1].startswith(("sm_", "sm-"))
]
structured_to_hub_edges = [
    edge for edge in all_edges
    if not edge[2]
    and "/structured/" in edge[0]
    and edge[1].startswith(("topics/", "files/", "profile/"))
]
assert chronology_edges == []
assert message_to_structured_edges
assert structured_to_hub_edges

mtimes = {
    path: path.stat().st_mtime_ns
    for path in (
        active_msg,
        forgotten_msg,
        second_active_msg,
        active_structured,
        code_structured,
        forgotten_structured,
        topic_mirror,
        topic_ragmemory,
        topic_c,
        topic_collision,
        file_hub,
        profile_hub,
        timeline,
        turns,
        OUT_PATH / "index.md",
    )
}
second = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2, config_path=CONFIG_PATH)
assert second["written"] == 0
assert second["removed"] == 0
assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    conn.execute("UPDATE messages SET tombstoned = 1 WHERE message_id = 1")

third = module.export_obsidian(DB_PATH, OUT_PATH, timeline_page_size=2, config_path=CONFIG_PATH)
assert third["removed"] >= 3
assert not active_msg.exists()
assert not active_structured.exists()
assert not code_structured.exists()
assert (OUT_PATH / "forgotten/messages/msg-000001.md").exists()
assert (OUT_PATH / "forgotten/structured/sm_active.md").exists()
assert (OUT_PATH / "forgotten/structured/sm_code.md").exists()

print("Obsidian export test passed.")
