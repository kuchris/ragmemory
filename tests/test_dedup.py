"""
Verify add_message skips normalized duplicate messages.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_dedup.py
"""
import shutil
from pathlib import Path

from ragmemory.memory import MemoryStore

DB_PATH = Path("./.data/chroma_dedup_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))

first = store.add_message(
    "user",
    "Remember the API contract returns structured data.",
    extract_structured=False,
)
duplicate = store.add_message(
    "user",
    "  Remember   the API contract returns structured data.  ",
    extract_structured=False,
)
same_text_other_role = store.add_message(
    "assistant",
    "Remember the API contract returns structured data.",
    extract_structured=False,
)

assert first.saved is True
assert first.deduped is False
assert first.message_id == 0
assert len(first.chunk_ids) == 1

assert duplicate.saved is False
assert duplicate.deduped is True
assert duplicate.message_id == first.message_id
assert duplicate.content_hash == first.content_hash
assert duplicate.chunk_ids == []

assert same_text_other_role.saved is True
assert same_text_other_role.deduped is False
assert same_text_other_role.message_id == 1

assert len(store.raw_log) == 2
assert store.collection.count() == 2

print("Dedup test passed.")
