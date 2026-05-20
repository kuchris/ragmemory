import os
import configparser
import re
import contextlib
import io
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from mcp.server.fastmcp import FastMCP
from ragmemory import MemoryStore, format_for_prompt
from scripts.export_obsidian import export_obsidian


DEFAULT_DB_PATH = ROOT_PATH / ".data" / "chroma_db"
DB_PATH = Path(os.environ.get("RAGMEMORY_DB_PATH", str(DEFAULT_DB_PATH)))
DEFAULT_OBSIDIAN_PATH = ROOT_PATH / ".data" / "obsidian_memory"
OBSIDIAN_PATH = Path(os.environ.get("RAGMEMORY_OBSIDIAN_PATH", str(DEFAULT_OBSIDIAN_PATH)))
FORGET_SAMPLE_LIMIT = 50
FORGET_TEXT_LIMIT = 180
REMOVE_MAX_IDS = 3
REMOVE_REASON_MIN_CHARS = 16
LOCAL_CONFIG_PATH = ROOT_PATH / "ragmemory.local.ini"

mcp = FastMCP("RAG Memory")
with contextlib.redirect_stdout(io.StringIO()):
    store = MemoryStore(db_path=str(DB_PATH))


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def build_recall_context(user_message: str) -> str:
    quiet_call(store.configure_recall, **hook_recall_settings())
    bundle = quiet_call(store.build_context_bundle, user_message)
    quiet_call(store.commit_drops, bundle)
    return format_for_prompt(bundle)


def save_user_message(user_message: str, extract_structured: bool | str = "background"):
    return quiet_call(
        store.add_message,
        "user",
        user_message,
        extract_structured=extract_structured,
    )


def save_assistant_message(message: str, extract_structured: bool | str = "background"):
    return quiet_call(
        store.add_message,
        "assistant",
        message,
        extract_structured=extract_structured,
    )


def export_obsidian_mirror() -> str:
    try:
        stats = export_obsidian(DB_PATH, OBSIDIAN_PATH)
    except Exception as exc:
        return f"Obsidian export failed: {exc}"
    return (
        f"Obsidian mirror updated: {OBSIDIAN_PATH} "
        f"(written {stats['written']}, removed {stats['removed']})."
    )


def run_pending_structured_extractions(limit: int = 3) -> list[str]:
    return quiet_call(store.run_pending_extractions, limit=limit)


def stats() -> dict:
    return {
        "db_path": str(DB_PATH),
        "obsidian_path": str(OBSIDIAN_PATH),
        "chunks": store.collection.count(),
        "next_message_id": store.message_id,
        "messages": len(store.raw_log),
        "bm25_indexed": len(store.bm25),
        "ledger_entries": len(store.ledger),
        "structured_objects": len(store.structured),
        "pending_extractions": len(store._pending_extractions),
    }


def _config_bool(section: str, key: str, default: bool = False) -> bool:
    env_key = f"RAGMEMORY_{section}_{key}".upper().replace(".", "_")
    if env_key in os.environ:
        return os.environ[env_key].strip().lower() in {"1", "true", "yes", "on"}
    parser = configparser.ConfigParser()
    parser.read(LOCAL_CONFIG_PATH, encoding="utf-8-sig")
    if not parser.has_section(section) or key not in parser[section]:
        return default
    return parser[section].getboolean(key, fallback=default)


def _config_int(section: str, key: str, default: int) -> int:
    env_key = f"RAGMEMORY_{section}_{key}".upper().replace(".", "_")
    if env_key in os.environ:
        try:
            return max(0, int(os.environ[env_key].strip()))
        except ValueError:
            return default
    parser = configparser.ConfigParser()
    parser.read(LOCAL_CONFIG_PATH, encoding="utf-8-sig")
    if not parser.has_section(section) or key not in parser[section]:
        return default
    try:
        return max(0, parser[section].getint(key, fallback=default))
    except ValueError:
        return default


def hook_recall_settings() -> dict:
    return {
        "context_token_budget": _config_int("recall", "context_token_budget", 2000),
        "retrieve_top_k": _config_int("recall", "retrieve_top_k", 5),
        "structured_top_k": _config_int("recall", "structured_top_k", 3),
        "recent_messages": _config_int("recall", "recent_messages", 12),
        "include_recent": _config_bool("recall", "include_recent", default=True),
        "include_structured": _config_bool("recall", "include_structured", default=True),
    }


def tombstone_enabled() -> bool:
    return _config_bool("mcp.tools", "enable_tombstone", default=False)


def mcp_recall_enabled() -> bool:
    return _config_bool("mcp.tools", "enable_recall", default=False)


def mcp_save_enabled() -> bool:
    return _config_bool("mcp.tools", "enable_save", default=False)


def _clip(text: str, limit: int = FORGET_TEXT_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _parse_message_ids(message_ids: str) -> list[int] | None:
    if not message_ids.strip():
        return None

    ids = []
    for part in re.split(r"[,\s]+", message_ids.strip()):
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid message id: {part}") from exc

    return ids or None


def _forget_kwargs(message_ids: str, before: str, sample_limit: int) -> dict:
    ids = _parse_message_ids(message_ids)
    cutoff = before.strip() or None
    if ids and cutoff:
        raise ValueError("Use message_ids or before, not both.")
    if not ids and not cutoff:
        raise ValueError("Provide message_ids or before.")
    return {
        "message_ids": ids,
        "before": cutoff,
        "sample_limit": sample_limit,
    }


def _remove_confirm_kwargs(message_ids: str, reason: str, sample_limit: int) -> tuple[dict, list[int], str]:
    ids = _parse_message_ids(message_ids)
    if not ids:
        raise ValueError("remove_memory_confirm requires explicit message_ids.")
    if len(ids) > REMOVE_MAX_IDS:
        raise ValueError(f"remove_memory_confirm accepts at most {REMOVE_MAX_IDS} message IDs.")
    normalized_reason = " ".join(reason.split())
    if len(normalized_reason) < REMOVE_REASON_MIN_CHARS:
        raise ValueError(f"reason must be at least {REMOVE_REASON_MIN_CHARS} non-whitespace characters.")
    return {
        "message_ids": ids,
        "before": None,
        "sample_limit": sample_limit,
    }, ids, normalized_reason


def _format_remove_preview(preview, *, confirmed: bool = False) -> str:
    title = "Remove confirmed" if confirmed else "Remove preview"
    lines = [
        f"{title}:",
        (
            f"Messages: {preview.message_count} | Chunks: {preview.chunk_count} | "
            f"Structured: {preview.structured_count} | Ledger: {preview.ledger_count} | "
            f"Truncated: {preview.truncated}"
        ),
    ]

    if confirmed:
        lines.append(f"Tombstoned messages: {preview.tombstoned_count} | Event: {preview.event_id}")
    else:
        lines.append("Run remove_memory_confirm with explicit message_ids and a reason to tombstone records.")

    if preview.messages:
        lines.append("")
        lines.append("Message samples:")
        for message in preview.messages:
            lines.append(f"- {message.message_id} ({message.role}): {_clip(message.text)}")

    if preview.chunks:
        lines.append("")
        lines.append("Chunk samples:")
        for chunk in preview.chunks:
            lines.append(f"- {chunk.id} from message {chunk.message_id}: {_clip(chunk.text)}")

    if preview.structured:
        lines.append("")
        lines.append("Structured samples:")
        for obj in preview.structured:
            lines.append(f"- {obj.id} ({obj.type}) from message {obj.message_id}: {_clip(obj.summary)}")

    if preview.ledger_entries:
        lines.append("")
        lines.append("Ledger samples:")
        for entry in preview.ledger_entries:
            lines.append(f"- {entry.chunk_id} from message {entry.message_id}: {_clip(entry.text)}")

    return "\n".join(lines)


def _log_mcp_event(event: str, **payload) -> str:
    return quiet_call(store._log_event, event, **payload)


@mcp.tool()
def recall(user_message: str) -> str:
    """Return relevant memory context only when [mcp.tools] enable_recall = true. Hooks should own recall when installed."""
    if not mcp_recall_enabled():
        return "MCP recall disabled: hook-based recall is active. Set [mcp.tools] enable_recall = true in ragmemory.local.ini to enable."
    context = build_recall_context(user_message)
    if mcp_save_enabled():
        save_user_message(user_message, extract_structured="background")
        export_obsidian_mirror()
    return context if context else "No relevant memory found."


@mcp.tool()
def save(summary: str) -> str:
    """Call at the END of every turn. Store a short summary of your response."""
    if not mcp_save_enabled():
        return "MCP save disabled: set [mcp.tools] enable_save = true in ragmemory.local.ini."
    save_assistant_message(summary)
    created = run_pending_structured_extractions(limit=1)
    export_status = export_obsidian_mirror()
    if created:
        return f"Saved. Structured objects created: {len(created)}. {export_status}"
    return f"Saved. {export_status}"


@mcp.tool()
def remember_document(text: str, role: str = "user") -> str:
    """Store a large document or long text by splitting it into chunks first."""
    if not mcp_save_enabled():
        return "MCP save disabled: set [mcp.tools] enable_save = true in ragmemory.local.ini."
    normalized = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)
    result = quiet_call(store.add_message, role, normalized, extract_structured="background")
    export_status = export_obsidian_mirror()
    return (
        f"Stored {len(result.chunk_ids)} chunk(s). "
        f"Queued jobs: {len(result.queued_job_ids)}. "
        f"Total chunks: {store.collection.count()}. "
        f"{export_status}"
    )


@mcp.tool()
def remove_memory_preview(message_ids: str = "", before: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Preview records that would be tombstoned. Use only when the user explicitly asks to forget/remove/delete memory."""
    try:
        kwargs = _forget_kwargs(message_ids, before, sample_limit)
        preview = quiet_call(store.forget, **kwargs)
    except (NotImplementedError, ValueError) as exc:
        return f"Remove preview failed: {exc}"
    return _format_remove_preview(preview)


@mcp.tool()
def remove_memory_confirm(message_ids: str = "", reason: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Tombstone explicit message_ids only after user explicitly asks to forget/remove/delete them. Automatic decay handles staleness."""
    if not tombstone_enabled():
        return "Remove confirm disabled: set [mcp.tools] enable_tombstone = true in ragmemory.local.ini."
    try:
        kwargs, ids, normalized_reason = _remove_confirm_kwargs(message_ids, reason, sample_limit)
        result = quiet_call(store.forget, **kwargs, confirm=True)
    except (NotImplementedError, ValueError) as exc:
        return f"Remove confirm failed: {exc}"
    _log_mcp_event(
        "memory_tombstoned_via_mcp",
        message_ids=ids,
        reason=normalized_reason,
        turn_id=max(store.message_id - 1, 0),
        client_info="mcp",
        tombstone_event_id=result.event_id,
    )
    return _format_remove_preview(result, confirmed=True) + "\n\n" + export_obsidian_mirror()


@mcp.tool()
def forget_preview(message_ids: str = "", before: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Deprecated alias for remove_memory_preview. Will be removed once no callers remain."""
    _log_mcp_event(
        "mcp_tool_deprecated_call",
        tool="forget_preview",
        replacement="remove_memory_preview",
        client_info="mcp",
    )
    return remove_memory_preview(message_ids=message_ids, before=before, sample_limit=sample_limit)


@mcp.tool()
def forget_confirm(message_ids: str = "", before: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Deprecated alias for remove_memory_confirm. Will be removed once no callers remain."""
    _log_mcp_event(
        "mcp_tool_deprecated_call",
        tool="forget_confirm",
        replacement="remove_memory_confirm",
        client_info="mcp",
    )
    if before.strip():
        return "Forget confirm failed: before= is preview-only. Use remove_memory_preview(before=...), then confirm explicit message_ids."
    return remove_memory_confirm(
        message_ids=message_ids,
        reason="Deprecated forget_confirm alias used after explicit user approval.",
        sample_limit=sample_limit,
    )


@mcp.tool()
def memory_stats() -> str:
    """Get current memory statistics."""
    s = stats()
    return (
        f"DB: {s['db_path']} | Chunks: {s['chunks']} | "
        f"Obsidian: {s['obsidian_path']} | "
        f"Next message ID: {s['next_message_id']} | Messages: {s['messages']} | "
        f"Structured: {s['structured_objects']} | Pending extraction jobs: {s['pending_extractions']} | "
        f"Ledger: {s['ledger_entries']}"
    )


if __name__ == "__main__":
    mcp.run()
