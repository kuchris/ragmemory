"""
Verify Stop hook file-change summaries.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_stop_hook_changes.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragm_mcp.hooks.stop import (
    _extract_tool_actions_from_transcript,
    assistant_message_from_payload,
)


def write_transcript(path: Path, items: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(item) for item in items),
        encoding="utf-8",
    )


patch_text = """*** Begin Patch
*** Update File: src/ragmemory/memory.py
@@
+changed
*** Add File: tests/test_stop_hook_changes.py
+new
*** Move to: docs/new-name.md
*** End Patch
"""

with tempfile.TemporaryDirectory() as tmp:
    transcript = Path(tmp) / "transcript.jsonl"
    write_transcript(
        transcript,
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old turn"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": "*** Begin Patch\n*** Update File: old.py\n*** End Patch\n",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "latest turn"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": patch_text,
                },
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "edit",
                        "input": {"path": "ragm_mcp/hooks/stop.py"},
                    },
                    {
                        "type": "tool_use",
                        "name": "write",
                        "input": {},
                    },
                ],
            },
        ],
    )

    actions = _extract_tool_actions_from_transcript(str(transcript))
    assert {"tool": "apply_patch", "path": "src/ragmemory/memory.py"} in actions
    assert {
        "tool": "apply_patch",
        "path": "tests/test_stop_hook_changes.py",
    } in actions
    assert {"tool": "apply_patch", "path": "docs/new-name.md"} in actions
    assert {"tool": "edit", "path": "ragm_mcp/hooks/stop.py"} in actions
    assert all(action["path"] != "old.py" for action in actions)
    assert all(action["path"] != "unknown" for action in actions)

    message = assistant_message_from_payload(
        {
            "last_assistant_message": "Done.",
            "transcript_path": str(transcript),
        }
    )
    assert "## Files changed" in message
    assert "- src/ragmemory/memory.py (apply_patch)" in message
    assert "- ragm_mcp/hooks/stop.py (edit)" in message
    assert "old.py" not in message
    assert "unknown" not in message

print("Stop hook change extraction test passed.")
