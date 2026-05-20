"""
Preview and confirm RagMemory removals from the command line.

This is the manual safety tool for wrong, private, or harmful memories. It
tombstones selected messages; it is not the automatic forgetting/decay system.

Examples:
    uv run python scripts/remove_memory.py --recent 20
    uv run python scripts/remove_memory.py --search "wrong project path"
    uv run python scripts/remove_memory.py --message-ids 12,13
    uv run python scripts/remove_memory.py --message-ids 12,13 --confirm
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragmemory import MemoryStore
from scripts.export_obsidian import export_obsidian


DEFAULT_DB_PATH = Path(os.environ.get("RAGMEMORY_DB_PATH", "./.data/chroma_db"))
DEFAULT_OBSIDIAN_PATH = Path(os.environ.get("RAGMEMORY_OBSIDIAN_PATH", "./.data/obsidian_memory"))
TEXT_LIMIT = 180


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def parse_message_ids(value: str) -> list[int] | None:
    if not value.strip():
        return None
    ids = []
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        ids.append(int(part))
    return ids or None


def quiet_store(db_path: Path) -> MemoryStore:
    with contextlib.redirect_stdout(io.StringIO()):
        return MemoryStore(db_path=str(db_path))


def print_recent(store: MemoryStore, limit: int) -> None:
    messages = store.raw_log[-max(limit, 0):]
    if not messages:
        print("No active messages.")
        return
    for message in messages:
        print(
            f"{message['message_id']} | {message['role']} | "
            f"{message.get('created_at', '')} | {clip(message['text'])}"
        )


def print_search(store: MemoryStore, query: str, limit: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        results = store.search(query, top_k=limit)
    if not results:
        print("No active matches.")
        return
    seen = set()
    for result in results:
        key = (result.message_id, result.source, result.item_id)
        if key in seen:
            continue
        seen.add(key)
        print(
            f"{result.message_id} | {result.source} | score={result.score:.4f} | "
            f"{clip(result.text)}"
        )


def format_preview(preview, confirmed: bool) -> str:
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
        lines.append("Preview only. Re-run with --confirm to tombstone these records.")

    if preview.messages:
        lines.extend(["", "Message samples:"])
        for message in preview.messages:
            lines.append(f"- {message.message_id} ({message.role}): {clip(message.text)}")
    if preview.chunks:
        lines.extend(["", "Chunk samples:"])
        for chunk in preview.chunks:
            lines.append(f"- {chunk.id} from message {chunk.message_id}: {clip(chunk.text)}")
    if preview.structured:
        lines.extend(["", "Structured samples:"])
        for obj in preview.structured:
            lines.append(f"- {obj.id} ({obj.type}) from message {obj.message_id}: {clip(obj.summary)}")
    if preview.ledger_entries:
        lines.extend(["", "Ledger samples:"])
        for entry in preview.ledger_entries:
            lines.append(f"- {entry.chunk_id} from message {entry.message_id}: {clip(entry.text)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find, preview, and tombstone wrong/private/harmful RagMemory records."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--obsidian-output", type=Path, default=DEFAULT_OBSIDIAN_PATH)
    parser.add_argument("--message-ids", default="", help="Comma/space separated message IDs.")
    parser.add_argument("--before", default="", help="Tombstone messages before this ISO timestamp.")
    parser.add_argument("--confirm", action="store_true", help="Apply the tombstone after previewing.")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--recent", type=int, help="List recent active messages and exit.")
    parser.add_argument("--search", help="Search active memory for candidate message IDs and exit.")
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--no-export", action="store_true", help="Do not update the Obsidian mirror after confirm.")
    args = parser.parse_args(argv)

    store = quiet_store(args.db_path)

    if args.recent is not None:
        print_recent(store, args.recent)
        return 0
    if args.search:
        print_search(store, args.search, args.search_limit)
        return 0

    message_ids = parse_message_ids(args.message_ids)
    before = args.before.strip() or None
    if message_ids and before:
        parser.error("Use --message-ids or --before, not both.")
    if not message_ids and not before:
        parser.error("Provide --message-ids/--before, or use --recent/--search to find candidates.")

    with contextlib.redirect_stdout(io.StringIO()):
        result = store.forget(
            message_ids=message_ids,
            before=before,
            sample_limit=args.sample_limit,
            confirm=args.confirm,
        )
    print(format_preview(result, confirmed=args.confirm))

    if args.confirm and not args.no_export:
        stats = export_obsidian(args.db_path, args.obsidian_output)
        print(
            f"\nObsidian mirror updated: {args.obsidian_output} "
            f"(written {stats['written']}, removed {stats['removed']})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
