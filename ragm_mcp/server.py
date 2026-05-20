import os
import re
import contextlib
import io
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from ragmemory import MemoryStore, format_for_prompt


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / ".data" / "chroma_db"
DB_PATH = Path(os.environ.get("RAGMEMORY_DB_PATH", str(DEFAULT_DB_PATH)))

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


def stats() -> dict:
    return {
        "db_path": str(DB_PATH),
        "chunks": store.collection.count(),
        "next_message_id": store.message_id,
        "messages": len(store.raw_log),
        "bm25_indexed": len(store.bm25),
        "ledger_entries": len(store.ledger),
        "structured_objects": len(store.structured),
        "pending_extractions": len(store._pending_extractions),
    }


@mcp.tool()
def recall(user_message: str) -> str:
    """Call at the START of every turn. Stores the user message and returns relevant memory context."""
    context = build_recall_context(user_message)
    save_user_message(user_message, extract_structured="background")
    return context if context else "No relevant memory found."


@mcp.tool()
def save(summary: str) -> str:
    """Call at the END of every turn. Store a short summary of your response."""
    save_assistant_message(summary)
    created = quiet_call(store.run_pending_extractions, limit=1)
    if created:
        return f"Saved. Structured objects created: {len(created)}."
    return "Saved."


@mcp.tool()
def remember_document(text: str, role: str = "user") -> str:
    """Store a large document or long text by splitting it into chunks first."""
    normalized = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)
    result = quiet_call(store.add_message, role, normalized, extract_structured="background")
    return (
        f"Stored {len(result.chunk_ids)} chunk(s). "
        f"Queued jobs: {len(result.queued_job_ids)}. "
        f"Total chunks: {store.collection.count()}"
    )


@mcp.tool()
def memory_stats() -> str:
    """Get current memory statistics."""
    s = stats()
    return (
        f"DB: {s['db_path']} | Chunks: {s['chunks']} | "
        f"Next message ID: {s['next_message_id']} | Messages: {s['messages']} | "
        f"Structured: {s['structured_objects']} | Pending extraction jobs: {s['pending_extractions']} | "
        f"Ledger: {s['ledger_entries']}"
    )


if __name__ == "__main__":
    mcp.run()
