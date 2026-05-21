"""
Verify message compaction stays side-by-side with raw messages.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_message_compaction.py
"""
import json
import shutil
import sqlite3
from pathlib import Path

from ragmemory import MemoryStore


DB_PATH = Path("./.data/chroma_message_compaction_test")
EVENTS_PATH = DB_PATH / "events.jsonl"


class FakeCompactor:
    def __init__(self, compact_text: str | None):
        self.compact_text = compact_text
        self.last_error = "fake failure" if compact_text is None else None
        self.calls = 0

    def compact(self, role: str, text: str, target_ratio: float, evidence_refs=None) -> str | None:
        self.calls += 1
        return self.compact_text


def compact_columns(message_id: int) -> tuple:
    with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
        return conn.execute(
            """
            SELECT text, compact_text, compact_status, compact_model
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()


if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
store.compaction_options.enabled = True
store.compaction_options.model = "fake-compact-model"
store.compaction_options.min_chars = 80
store.compaction_options.max_chars = 10000
store.compaction_options.mode = "background"

raw = "\n".join([
    "IDE context filler " * 80,
    "Added preservation checks for paths, commands, URLs, backticked tokens, and error lines.",
    "See [server.py](/c:/Users/alten/Desktop/ku/ragmemory/ragm_mcp/server.py:78).",
    "C:\\foo\\bar.ini",
    "pwsh -c \"uv run python scripts\\compact_backfill.py --limit 20\"",
    "ValueError: bad config key `RAGMEMORY_COMPACT_ENABLE`",
    "User prefers compact messages in background mode.",
])
compact = "\n".join([
    "User prefers compact messages in background mode.",
    "/c:/Users/alten/Desktop/ku/ragmemory/ragm_mcp/server.py",
    "C:\\foo\\bar.ini",
    "pwsh -c \"uv run python scripts\\compact_backfill.py --limit 20\"",
    "ValueError: bad config key `RAGMEMORY_COMPACT_ENABLE`",
    "RAGMEMORY_COMPACT_ENABLE",
])
store.compactor = FakeCompactor(compact)
result = store.add_message("user", raw, extract_structured=False)

assert result.saved is True
assert len(result.queued_job_ids) == 1
assert store.job_counts().get("pending", 0) == 1
assert store.run_pending_compactions(limit=0) == []
assert store.run_pending_compactions(limit=1) == [result.message_id]

stored_raw, stored_compact, status, model = compact_columns(result.message_id)
assert stored_raw == raw
assert stored_compact == compact
assert status == "ok"
assert model == "fake-compact-model"
search_results = store.search("RAGMEMORY_COMPACT_ENABLE", top_k=3)
assert any(
    item.message_id == result.message_id
    and item.source == "compact_chunk"
    and "User prefers compact messages in background mode." in item.text
    for item in search_results
)
assert all("IDE context filler" not in item.text for item in search_results)
assert store.compact_existing_messages(limit=10) == []

warning_raw = "\n".join([
    "# Task Group: misc",
    "## Reusable knowledge",
    "This compact warning fixture has no critical file paths or errors. " * 10,
    "```text\nplain explanatory block can live in structured memory\n```",
    "The user mentioned `RAGMEMORY_WARNING_TOKEN` in passing.",
])
warning_compact = "This compact warning fixture has no critical file paths or errors."
store.compactor = FakeCompactor(warning_compact)
warning = store.add_message("user", warning_raw, extract_structured=False)
assert store.run_pending_compactions(limit=1) == [warning.message_id]
_, stored_compact, status, _ = compact_columns(warning.message_id)
assert stored_compact == warning_compact
assert status == "ok"

block_raw = "\n".join([
    "Use a stable evidence ref for this exact C# block. " * 4,
    "```csharp",
    "private const int MarsHost = 24;",
    "if (unit.UnitModelCode == MarsHost)",
    "{",
    "    temp[0] += \"0101\";",
    "}",
    "```",
])
block_ref = store._block_evidence_references(block_raw)[0]
same_block_ref = store._block_evidence_references(block_raw)[0]
assert block_ref.marker == same_block_ref.marker
assert block_ref.marker.startswith("evidence[csharp:")
store.compactor = FakeCompactor(
    f"Use MarsHost fixed values for Mars-only ID generation. See {block_ref.marker}."
)
block_result = store.add_message("user", block_raw, extract_structured=False)
assert store.run_pending_compactions(limit=1) == [block_result.message_id]
_, stored_compact, status, _ = compact_columns(block_result.message_id)
assert block_ref.marker in stored_compact
assert status == "ok"

store.compactor = FakeCompactor("Hallucinated reference evidence[csharp:deadbeefdead].")
invalid_ref = store.add_message("user", block_raw + "\ninvalid ref case", extract_structured=False)
assert store.run_pending_compactions(limit=1) == []
_, compact_text, status, _ = compact_columns(invalid_ref.message_id)
assert compact_text is None
assert status == "failed"

store.compactor = FakeCompactor("Malformed reference evidence[csharp:deadbeef].")
malformed_ref = store.add_message("user", block_raw + "\nmalformed ref case", extract_structured=False)
assert store.run_pending_compactions(limit=1) == []
_, compact_text, status, _ = compact_columns(malformed_ref.message_id)
assert compact_text is None
assert status == "failed"

literal_ref_raw = (
    "This message is discussing evidence refs as literal text, like "
    "evidence[python:6fda8887e72a]. It does not contain the original block. " * 4
)
store.compactor = FakeCompactor(
    "Discusses literal evidence ref text evidence[python:6fda8887e72a]."
)
literal_ref = store.add_message("assistant", literal_ref_raw, extract_structured=False)
assert store.run_pending_compactions(limit=1) == [literal_ref.message_id]
_, stored_compact, status, _ = compact_columns(literal_ref.message_id)
assert stored_compact == "Discusses literal evidence ref text evidence[python:6fda8887e72a]."
assert status == "ok"

normalized_raw = "Normalization should accept /foo/bar.py when compact text keeps foo/bar.py. " * 3
normalized_compact = "Normalization keeps foo/bar.py."
store.compactor = FakeCompactor(normalized_compact)
normalized = store.add_message("user", normalized_raw, extract_structured=False)
assert store.run_pending_compactions(limit=1) == [normalized.message_id]
_, stored_compact, status, _ = compact_columns(normalized.message_id)
assert stored_compact == normalized_compact
assert status == "ok"

short_fake = FakeCompactor("should not be used")
store.compactor = short_fake
short = store.add_message("user", "tiny", extract_structured=False)
_, compact_text, status, _ = compact_columns(short.message_id)
assert compact_text is None
assert status == "skipped_short"
assert short_fake.calls == 0

too_long_fake = FakeCompactor("should not be used")
store.compactor = too_long_fake
store.compaction_options.max_chars = 120
too_long = store.add_message("user", "too long message " * 20, extract_structured=False)
assert store.run_pending_compactions(limit=1) == []
_, compact_text, status, _ = compact_columns(too_long.message_id)
assert compact_text is None
assert status == "too_long"
assert too_long_fake.calls == 0
store.compaction_options.max_chars = 10000

store.compactor = FakeCompactor(None)
failure = store.add_message("user", raw + "\nAPI failure case", extract_structured=False)
assert store.run_pending_compactions(limit=1) == []
stored_raw, compact_text, status, _ = compact_columns(failure.message_id)
assert stored_raw.endswith("API failure case")
assert compact_text is None
assert status == "failed"

store.compactor = FakeCompactor("Summary missing the required path.")
missing = store.add_message("user", raw + "\nMissing evidence case", extract_structured=False)
assert store.run_pending_compactions(limit=1) == []
_, compact_text, status, _ = compact_columns(missing.message_id)
assert compact_text is None
assert status == "failed"

events = [
    json.loads(line)
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
event_names = [event["event"] for event in events]
assert "message_compaction_queued" in event_names
assert "message_compacted" in event_names
assert "message_compacted_with_warnings" in event_names
assert "message_compaction_failed" in event_names
assert "message_compaction_skipped" in event_names

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    conn.execute(
        """
        UPDATE messages
        SET compact_text = ?, compact_status = 'ok'
        WHERE message_id = ?
        """,
        (
            "REBUILT_COMPACT_ONLY C:\\foo\\bar.ini RAGMEMORY_COMPACT_ENABLE",
            result.message_id,
        ),
    )
assert store.rebuild_chat_memory_index() > 0
rebuilt_results = store.search("REBUILT_COMPACT_ONLY", top_k=3)
assert any(
    item.message_id == result.message_id
    and item.source == "compact_chunk"
    and "REBUILT_COMPACT_ONLY" in item.text
    for item in rebuilt_results
)

print("Message compaction test passed.")
