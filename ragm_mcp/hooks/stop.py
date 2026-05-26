import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def assistant_message_from_payload(payload: dict) -> str:
    for key in ("last_assistant_message", "assistant_message", "response", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _append_changes(value.strip(), payload)
    return assistant_message_from_transcript(payload)


def assistant_message_from_transcript(payload: dict) -> str:
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path.strip():
        return ""

    transcript = Path(path)
    if not transcript.exists():
        return ""

    last = ""
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = find_assistant_text(item)
        if candidate:
            last = candidate
    if not last:
        return ""
    return _append_changes(last.strip(), payload)


def find_assistant_text(value) -> str:
    if isinstance(value, dict):
        role = value.get("role") or value.get("author") or value.get("type")
        if role == "assistant":
            text = text_from_value(value.get("content") or value.get("text") or value.get("message"))
            if text:
                return text
        for child in value.values():
            text = find_assistant_text(child)
            if text:
                return text
    if isinstance(value, list):
        for child in value:
            text = find_assistant_text(child)
            if text:
                return text
    return ""


def text_from_value(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [text_from_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return text_from_value(value.get("text") or value.get("content"))
    return ""


# Change extraction

FILE_MODIFY_TOOLS = {"edit", "write", "ast_edit", "ast_grep_edit"}
APPLY_PATCH_TOOLS = {"apply_patch", "functions.apply_patch"}
PATCH_FILE_PREFIXES = (
    "*** Add File: ",
    "*** Update File: ",
    "*** Delete File: ",
    "*** Move to: ",
)


def _extract_tool_actions_from_transcript(transcript_path: str) -> list[dict]:
    """Parse transcript JSONL and extract file-modifying tool calls."""
    path = Path(transcript_path)
    if not path.exists():
        return []

    actions: list[dict] = []
    seen = set()
    items: list[dict] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        items.append(item)

    for item in _items_after_latest_user(items):
        for action in _extract_tool_calls_from_item(item):
            if not action.get("path"):
                continue
            key = (action["tool"], action["path"])
            if key not in seen:
                seen.add(key)
                actions.append(action)

    return actions


def _items_after_latest_user(items: list[dict]) -> list[dict]:
    latest_user_index = -1
    for index, item in enumerate(items):
        if _contains_user_message(item):
            latest_user_index = index
    return items[latest_user_index + 1:] if latest_user_index >= 0 else items


def _contains_user_message(value) -> bool:
    if isinstance(value, dict):
        if value.get("role") == "user":
            return True
        if value.get("type") == "user_message":
            return True
        return any(_contains_user_message(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_user_message(child) for child in value)
    return False


def _extract_tool_calls_from_item(value) -> list[dict]:
    """Recursively find file-modifying tool calls in a transcript item."""
    results: list[dict] = []

    if isinstance(value, dict):
        if value.get("type") == "function_call":
            results.extend(_actions_from_function_call(value))

        content = value.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    results.extend(_actions_from_tool_use(block))

        for child in value.values():
            results.extend(_extract_tool_calls_from_item(child))

    if isinstance(value, list):
        for child in value:
            results.extend(_extract_tool_calls_from_item(child))

    return results


def _actions_from_tool_use(block: dict) -> list[dict]:
    tool = block.get("name", "")
    if tool in FILE_MODIFY_TOOLS:
        inp = block.get("input") or {}
        return [{"tool": tool, "path": _extract_path(inp)}]
    if tool in APPLY_PATCH_TOOLS:
        return _actions_from_patch_text(block.get("input"))
    return []


def _actions_from_function_call(call: dict) -> list[dict]:
    tool = call.get("name", "")
    arguments = call.get("arguments")
    if tool in APPLY_PATCH_TOOLS:
        return _actions_from_patch_text(arguments)
    if tool in FILE_MODIFY_TOOLS:
        inp = _parse_call_arguments(arguments)
        return [{"tool": tool, "path": _extract_path(inp)}]
    return []


def _parse_call_arguments(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _actions_from_patch_text(value) -> list[dict]:
    if isinstance(value, dict):
        text = value.get("patch") or value.get("input") or value.get("text") or ""
    else:
        text = value or ""
    if not isinstance(text, str):
        return []

    actions = []
    for line in text.splitlines():
        for prefix in PATCH_FILE_PREFIXES:
            if line.startswith(prefix):
                path = line[len(prefix):].strip()
                if path:
                    actions.append({"tool": "apply_patch", "path": path})
                break
    return actions


def _extract_path(inp: dict) -> str:
    """Extract target file path from tool input dict."""
    if not isinstance(inp, dict):
        return ""
    for key in ("path", "file_path", "file", "target"):
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _format_change_summary(actions: list[dict]) -> str:
    if not actions:
        return ""

    by_file: dict[str, set[str]] = {}
    for action in actions:
        by_file.setdefault(action["path"], set()).add(action["tool"])

    lines = ["## Files changed"]
    for path, tools in sorted(by_file.items()):
        lines.append(f"- {path} ({', '.join(sorted(tools))})")
    return "\n".join(lines)


def _append_changes(message: str, payload: dict) -> str:
    """Append file change summary to the assistant message."""
    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path.strip():
        return message

    actions = _extract_tool_actions_from_transcript(transcript_path)
    summary = _format_change_summary(actions)
    if not summary:
        return message

    return message + "\n\n" + summary


# Debug / main

def write_debug(payload: dict, saved: bool) -> None:
    root = Path.cwd()
    sys.path.insert(0, str(root))

    from ragm_mcp.server import DB_PATH

    debug_dir = DB_PATH
    debug_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "Stop",
        "keys": sorted(payload.keys()),
        "has_last_assistant_message": bool(payload.get("last_assistant_message")),
        "has_transcript_path": bool(payload.get("transcript_path")),
        "saved": saved,
    }
    with (debug_dir / "hook_debug.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    payload = read_payload()
    message = assistant_message_from_payload(payload)
    if message:
        root = Path.cwd()
        sys.path.insert(0, str(root))

        from ragm_mcp.server import (
            export_obsidian_mirror,
            save_assistant_message,
        )

        save_assistant_message(message, extract_structured="background")
        export_obsidian_mirror()

    write_debug(payload, bool(message))
    sys.stdout.write(json.dumps({}))


if __name__ == "__main__":
    main()
