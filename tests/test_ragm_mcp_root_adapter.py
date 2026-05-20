"""
Verify ragm_mcp uses the root ragmemory API without polluting stdout.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_ragm_mcp_root_adapter.py
"""
import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DB_PATH = Path("./.data/ragm_mcp_root_adapter_test")
OBSIDIAN_PATH = Path("./.data/ragm_mcp_root_adapter_obsidian_test")
ROOT = Path.cwd()

for path in (DB_PATH, OBSIDIAN_PATH):
    if path.exists():
        shutil.rmtree(path)

os.environ["RAGMEMORY_DB_PATH"] = str(DB_PATH)
os.environ["RAGMEMORY_OBSIDIAN_PATH"] = str(OBSIDIAN_PATH)
os.environ["NVIDIA_API_KEY"] = ""
sys.path.insert(0, str(ROOT))

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    server = importlib.import_module("ragm_mcp.server")

assert stdout.getvalue() == ""
assert server.DB_PATH == DB_PATH

with contextlib.redirect_stdout(stdout):
    empty_context = server.build_recall_context("adapter test")
    result = server.save_user_message("adapter test user message", extract_structured=False)
    server.save_assistant_message("adapter test assistant message")

assert empty_context == ""
assert result.saved is True
assert server.stats()["messages"] == 2
assert (DB_PATH / "state.sqlite").exists()

env = os.environ.copy()
env["RAGMEMORY_DB_PATH"] = str(DB_PATH)
env["RAGMEMORY_OBSIDIAN_PATH"] = str(OBSIDIAN_PATH)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

prompt = subprocess.run(
    [sys.executable, "ragm_mcp/hooks/user_prompt_submit.py"],
    input=json.dumps({"prompt": "adapter hook prompt"}),
    text=True,
    capture_output=True,
    env=env,
    check=True,
)
assert "Loading embedding model" not in prompt.stdout
if prompt.stdout.strip():
    payload = json.loads(prompt.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"

stop = subprocess.run(
    [sys.executable, "ragm_mcp/hooks/stop.py"],
    input=json.dumps({
        "last_assistant_message": """adapter hook assistant should extract this config.

```json
{"adapter_hook": true}
```"""
    }),
    text=True,
    capture_output=True,
    env=env,
    check=True,
)
assert stop.stdout.strip() == "{}"
assert (DB_PATH / "hook_debug.jsonl").exists()
assert (OBSIDIAN_PATH / "index.md").exists()
assert (OBSIDIAN_PATH / "active/messages/msg-000002.md").exists()
assert (OBSIDIAN_PATH / "active/messages/msg-000003.md").exists()
assert (DB_PATH / "structured_memory.jsonl").exists()
assert any((OBSIDIAN_PATH / "active/structured").glob("*.md"))

print("RagM MCP root adapter test passed.")
