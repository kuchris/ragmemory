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
            return value.strip()
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
    return last.strip()


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

        from ragm_mcp.server import export_obsidian_mirror, save_assistant_message

        save_assistant_message(message)
        export_obsidian_mirror()

    write_debug(payload, bool(message))
    sys.stdout.write(json.dumps({}))


if __name__ == "__main__":
    main()
