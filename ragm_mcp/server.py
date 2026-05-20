import os
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

mcp = FastMCP("RAG Memory")
with contextlib.redirect_stdout(io.StringIO()):
    store = MemoryStore(db_path=str(DB_PATH))


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def build_recall_context(user_message: str) -> str:
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


def save_assistant_message(message: str):
    return quiet_call(
        store.add_message,
        "assistant",
        message,
        extract_structured=False,
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


def _format_forget_preview(preview, *, confirmed: bool = False) -> str:
    title = "Forget confirmed" if confirmed else "Forget preview"
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
        lines.append("Run forget_confirm with the same selector to tombstone these records.")

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


@mcp.tool()
def recall(user_message: str) -> str:
    """Call at the START of every turn. Stores the user message and returns relevant memory context."""
    context = build_recall_context(user_message)
    save_user_message(user_message, extract_structured="background")
    export_obsidian_mirror()
    return context if context else "No relevant memory found."


@mcp.tool()
def save(summary: str) -> str:
    """Call at the END of every turn. Store a short summary of your response."""
    save_assistant_message(summary)
    created = quiet_call(store.run_pending_extractions, limit=1)
    export_status = export_obsidian_mirror()
    if created:
        return f"Saved. Structured objects created: {len(created)}. {export_status}"
    return f"Saved. {export_status}"


@mcp.tool()
def remember_document(text: str, role: str = "user") -> str:
    """Store a large document or long text by splitting it into chunks first."""
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
def forget_preview(message_ids: str = "", before: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Preview memory records that would be tombstoned. message_ids is comma/space separated."""
    try:
        kwargs = _forget_kwargs(message_ids, before, sample_limit)
        preview = quiet_call(store.forget, **kwargs)
    except (NotImplementedError, ValueError) as exc:
        return f"Forget preview failed: {exc}"
    return _format_forget_preview(preview)


@mcp.tool()
def forget_confirm(message_ids: str = "", before: str = "", sample_limit: int = FORGET_SAMPLE_LIMIT) -> str:
    """Tombstone records selected by message_ids or before after previewing them."""
    try:
        kwargs = _forget_kwargs(message_ids, before, sample_limit)
        result = quiet_call(store.forget, **kwargs, confirm=True)
    except (NotImplementedError, ValueError) as exc:
        return f"Forget confirm failed: {exc}"
    return _format_forget_preview(result, confirmed=True) + "\n\n" + export_obsidian_mirror()


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
