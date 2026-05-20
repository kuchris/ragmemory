"""
Verify the command-line remove helper for hook-only users.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_remove_memory_script.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

from ragmemory import MemoryStore


DB_PATH = Path("./.data/remove_memory_script_test")
OUT_PATH = Path("./.data/remove_memory_script_obsidian_test")

for path in (DB_PATH, OUT_PATH):
    if path.exists():
        shutil.rmtree(path)

store = MemoryStore(db_path=str(DB_PATH))
target = store.add_message(
    "user",
    "REMOVE_SCRIPT_TARGET should be found, previewed, and tombstoned.",
    extract_structured=False,
)
keep = store.add_message(
    "user",
    "REMOVE_SCRIPT_KEEP should remain searchable.",
    extract_structured=False,
)
assert target.message_id is not None
assert keep.message_id is not None

base_cmd = [
    sys.executable,
    "scripts/remove_memory.py",
    "--db-path",
    str(DB_PATH),
    "--obsidian-output",
    str(OUT_PATH),
]

recent = subprocess.run(
    [*base_cmd, "--recent", "5"],
    text=True,
    capture_output=True,
    check=True,
)
assert "REMOVE_SCRIPT_TARGET" in recent.stdout
assert str(target.message_id) in recent.stdout

search = subprocess.run(
    [*base_cmd, "--search", "REMOVE_SCRIPT_TARGET"],
    text=True,
    capture_output=True,
    check=True,
)
assert "REMOVE_SCRIPT_TARGET" in search.stdout
assert str(target.message_id) in search.stdout

preview = subprocess.run(
    [*base_cmd, "--message-ids", str(target.message_id)],
    text=True,
    capture_output=True,
    check=True,
)
assert "Remove preview:" in preview.stdout
assert "Preview only" in preview.stdout
assert "REMOVE_SCRIPT_TARGET" in preview.stdout

confirm = subprocess.run(
    [*base_cmd, "--message-ids", str(target.message_id), "--confirm"],
    text=True,
    capture_output=True,
    check=True,
)
assert "Remove confirmed:" in confirm.stdout
assert "Tombstoned messages: 1" in confirm.stdout
assert "Obsidian mirror updated:" in confirm.stdout

reloaded = MemoryStore(db_path=str(DB_PATH))
assert all(chunk.message_id != target.message_id for chunk in reloaded.retrieve("REMOVE_SCRIPT_TARGET"))
assert any(chunk.message_id == keep.message_id for chunk in reloaded.retrieve("REMOVE_SCRIPT_KEEP"))
assert (OUT_PATH / "forgotten/messages" / f"msg-{target.message_id:06d}.md").exists()

print("Remove memory script test passed.")
