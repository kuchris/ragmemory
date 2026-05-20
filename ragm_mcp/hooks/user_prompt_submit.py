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


def prompt_from_payload(payload: dict) -> str:
    for key in ("prompt", "user_prompt", "message", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> None:
    payload = read_payload()
    prompt = prompt_from_payload(payload)
    if not prompt:
        return

    root = Path.cwd()
    sys.path.insert(0, str(root))

    from ragm_mcp.server import build_recall_context, save_user_message

    context = build_recall_context(prompt)
    save_user_message(prompt, extract_structured=False)
    if context:
        sys.stdout.write(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": (
                    "=== RagMemory Context ===\n"
                    f"{context}\n"
                    "=== End RagMemory Context ==="
                ),
            },
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
