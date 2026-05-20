"""
Verify preview-only forget by message_id.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_forget_preview.py
"""
import shutil
from pathlib import Path

from ragmemory import ForgetPreview, MemoryStore
from ragmemory.memory import RetrievedChunk

DB_PATH = Path("./.data/chroma_forget_preview_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
result = store.add_message(
    "user",
    """Forget preview should show affected records.

```json
{"mode": "preview-only", "selector": "message_ids"}
```""",
)
assert result.message_id is not None

store.ledger.log(
    RetrievedChunk(
        id="ledger-forget-preview",
        text="Ledger record connected to the forgotten message.",
        importance=0.5,
        message_id=result.message_id,
    )
)

preview = store.forget(message_ids=[result.message_id])

assert isinstance(preview, ForgetPreview)
assert preview.message_count == 1
assert preview.chunk_count >= 1
assert preview.structured_count == 1
assert preview.ledger_count == 1
assert preview.truncated is False

assert preview.messages[0].message_id == result.message_id
assert preview.chunks[0].message_id == result.message_id
assert preview.structured[0].message_id == result.message_id
assert preview.ledger_entries[0].message_id == result.message_id

capped = store.forget(message_ids=[result.message_id], sample_limit=0)
assert capped.message_count == 1
assert capped.chunk_count >= 1
assert capped.structured_count == 1
assert capped.ledger_count == 1
assert capped.messages == []
assert capped.chunks == []
assert capped.structured == []
assert capped.ledger_entries == []
assert capped.truncated is True

try:
    store.forget()
    raise AssertionError("forget without message_ids should fail")
except ValueError:
    pass

try:
    store.forget(message_ids=[result.message_id], query="preview")
    raise AssertionError("query selector should not be implemented yet")
except NotImplementedError:
    pass

try:
    store.forget(query="preview")
    raise AssertionError("query selector should not be implemented yet")
except NotImplementedError:
    pass

try:
    store.forget(before="2026-01-01T00:00:00Z")
    raise AssertionError("before selector should not be implemented yet")
except NotImplementedError:
    pass

print("Forget preview test passed.")
