"""
Export RagMemory state to an Obsidian-readable Markdown mirror.

The mirror is a generated one-way projection. Edit the DB through RagMemory,
then re-run this script; do not treat the Markdown files as source of truth.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("./.data/chroma_db")
DEFAULT_OUTPUT_PATH = Path("./.data/obsidian_memory")
TIMELINE_PAGE_SIZE = 500


@dataclass
class MessageRow:
    message_id: int
    role: str
    text: str
    content_hash: str
    created_at: str | None
    tombstoned: bool


@dataclass
class StructuredRow:
    id: str
    type: str
    summary: str
    source_text: str
    tags: list[str]
    importance: float
    message_id: int
    role: str


def load_messages(db_path: Path) -> list[MessageRow]:
    state_db = db_path / "state.sqlite"
    if not state_db.exists():
        return []
    with sqlite3.connect(state_db) as conn:
        rows = conn.execute(
            """
            SELECT message_id, role, text, content_hash, created_at, tombstoned
            FROM messages
            ORDER BY message_id
            """
        ).fetchall()
    return [
        MessageRow(
            message_id=message_id,
            role=role,
            text=text,
            content_hash=content_hash,
            created_at=created_at,
            tombstoned=bool(tombstoned),
        )
        for message_id, role, text, content_hash, created_at, tombstoned in rows
    ]


def load_structured(db_path: Path) -> list[StructuredRow]:
    path = db_path / "structured_memory.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            StructuredRow(
                id=str(item["id"]),
                type=str(item["type"]),
                summary=str(item["summary"]),
                source_text=str(item["source_text"]),
                tags=[str(tag) for tag in item.get("tags", [])],
                importance=float(item.get("importance", 0.0)),
                message_id=int(item["message_id"]),
                role=str(item["role"]),
            )
        )
    return rows


def message_stem(message_id: int) -> str:
    return f"msg-{message_id:06d}"


def structured_stem(object_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", object_id).strip("-") or "structured"


def frontmatter(fields: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = json.dumps(value, ensure_ascii=False)
        elif value is None:
            rendered = "null"
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def wikilink(stem: str) -> str:
    return f"[[{stem}]]"


def message_markdown(
    message: MessageRow,
    structured: list[StructuredRow],
    previous_message: MessageRow | None = None,
    next_message: MessageRow | None = None,
) -> str:
    related = [obj for obj in structured if obj.message_id == message.message_id]
    parts = [
        frontmatter(
            {
                "message_id": message.message_id,
                "role": message.role,
                "status": "forgotten" if message.tombstoned else "active",
                "created_at": message.created_at,
                "content_hash": message.content_hash,
                "structured_objects": [structured_stem(obj.id) for obj in related],
                "previous_message": message_stem(previous_message.message_id) if previous_message else None,
                "next_message": message_stem(next_message.message_id) if next_message else None,
            }
        ),
        "",
        f"# {message_stem(message.message_id)}",
        "",
        f"- Role: `{message.role}`",
        f"- Status: `{'forgotten' if message.tombstoned else 'active'}`",
    ]
    if message.created_at:
        parts.append(f"- Created: `{message.created_at}`")
    if previous_message:
        parts.append(f"- Previous: {wikilink(message_stem(previous_message.message_id))}")
    if next_message:
        parts.append(f"- Next: {wikilink(message_stem(next_message.message_id))}")
    if related:
        parts.append("- Structured: " + ", ".join(wikilink(structured_stem(obj.id)) for obj in related))
    parts.extend(["", "## Text", "", message.text.strip(), ""])
    return "\n".join(parts)


def structured_markdown(obj: StructuredRow, message_lookup: dict[int, MessageRow]) -> str:
    message = message_lookup.get(obj.message_id)
    status = "forgotten" if message and message.tombstoned else "active"
    parts = [
        frontmatter(
            {
                "structured_id": obj.id,
                "type": obj.type,
                "status": status,
                "message_id": obj.message_id,
                "role": obj.role,
                "importance": obj.importance,
                "tags": obj.tags,
            }
        ),
        "",
        f"# {structured_stem(obj.id)}",
        "",
        f"- Type: `{obj.type}`",
        f"- Status: `{status}`",
        f"- Source message: {wikilink(message_stem(obj.message_id))}",
        f"- Importance: `{obj.importance}`",
    ]
    if obj.tags:
        parts.append("- Tags: " + ", ".join(f"`{tag}`" for tag in obj.tags))
    parts.extend(["", "## Summary", "", obj.summary.strip(), "", "## Source Text", "", obj.source_text.strip(), ""])
    return "\n".join(parts)


def index_markdown(
    messages: list[MessageRow],
    structured: list[StructuredRow],
    timeline_page_size: int = TIMELINE_PAGE_SIZE,
) -> str:
    active_messages = [message for message in messages if not message.tombstoned]
    forgotten_messages = [message for message in messages if message.tombstoned]
    message_lookup = {message.message_id: message for message in messages}
    active_structured = [
        obj for obj in structured
        if not message_lookup.get(obj.message_id, MessageRow(obj.message_id, obj.role, "", "", None, True)).tombstoned
    ]
    forgotten_structured = [obj for obj in structured if obj not in active_structured]
    updated_at = max((message.created_at for message in messages if message.created_at), default=None)
    first_timeline = timeline_path(Path("."), 0, max(timeline_page_size, 1)).with_suffix("").as_posix()

    recent = active_messages[-10:]
    parts = [
        frontmatter(
            {
                "generated": True,
                "updated_at": updated_at,
                "source": "ragmemory",
            }
        ),
        "",
        "# RagMemory Mirror",
        "",
        "This folder is generated from the RagMemory DB. Edit RagMemory, not these Markdown files.",
        "",
        "## Counts",
        "",
        f"- Active messages: {len(active_messages)}",
        f"- Forgotten messages: {len(forgotten_messages)}",
        f"- Active structured objects: {len(active_structured)}",
        f"- Forgotten structured objects: {len(forgotten_structured)}",
        "",
        "## Recent Active Messages",
        "",
    ]
    if recent:
        for message in reversed(recent):
            preview = " ".join(message.text.split())[:120]
            parts.append(f"- {wikilink(message_stem(message.message_id))} `{message.role}` {preview}")
    else:
        parts.append("- No active messages.")
    parts.extend([
        "",
        "## Maps",
        "",
        f"- [[{first_timeline}]]",
        "- [[maps/turns]]",
        "",
        "## Folders",
        "",
        "- [[active/messages]]",
        "- [[active/structured]]",
        "- [[forgotten/messages]]",
        "- [[forgotten/structured]]",
        "",
    ])
    return "\n".join(parts)


def timeline_markdown(messages: list[MessageRow], page_index: int, page_size: int) -> str:
    start = page_index * page_size
    page = messages[start : start + page_size]
    range_start = start + 1
    range_end = start + len(page)
    parts = [
        frontmatter(
            {
                "generated": True,
                "source": "ragmemory",
                "map": "timeline",
                "range_start": range_start,
                "range_end": range_end,
            }
        ),
        "",
        f"# Timeline {range_start:04d}-{range_start + page_size - 1:04d}",
        "",
        "Active messages in message_id order. Forgotten messages are hidden from this timeline.",
        "",
    ]
    if page:
        for message in page:
            preview = " ".join(message.text.split())[:120]
            parts.append(f"- {wikilink(message_stem(message.message_id))} `{message.role}` {preview}")
    else:
        parts.append("- No active messages.")
    parts.append("")
    return "\n".join(parts)


def timeline_path(output_path: Path, page_index: int, page_size: int) -> Path:
    start = page_index * page_size + 1
    end = start + page_size - 1
    return output_path / "maps" / f"timeline-{start:04d}-{end:04d}.md"


def turn_groups(messages: list[MessageRow]) -> list[list[MessageRow]]:
    # Turn rule: one user message plus contiguous following assistant messages, until the next user message.
    turns: list[list[MessageRow]] = []
    current: list[MessageRow] = []
    for message in messages:
        if message.role == "user":
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
        else:
            turns.append([message])
    if current:
        turns.append(current)
    return turns


def turns_markdown(messages: list[MessageRow]) -> str:
    turns = turn_groups(messages)
    parts = [
        frontmatter(
            {
                "generated": True,
                "source": "ragmemory",
                "map": "turns",
                "turn_rule": "one user message plus contiguous following assistant messages until the next user message",
            }
        ),
        "",
        "# Turns",
        "",
        "Rule: one user message plus contiguous following assistant messages, until the next user message.",
        "",
    ]
    if not turns:
        parts.extend(["- No active turns.", ""])
        return "\n".join(parts)
    for index, turn in enumerate(turns, start=1):
        parts.append(f"## Turn {index}")
        for message in turn:
            preview = " ".join(message.text.split())[:120]
            parts.append(f"- {message.role}: {wikilink(message_stem(message.message_id))} {preview}")
        parts.append("")
    return "\n".join(parts)


def write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n")
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") == normalized:
        return False
    path.write_text(normalized, encoding="utf-8")
    return True


def clean_stale_markdown(root: Path, expected: set[Path]) -> int:
    removed = 0
    for folder in ("active/messages", "active/structured", "forgotten/messages", "forgotten/structured", "maps"):
        base = root / folder
        if not base.exists():
            continue
        for path in base.glob("*.md"):
            if path not in expected:
                path.unlink()
                removed += 1
    for folder in ("active/messages", "active/structured", "forgotten/messages", "forgotten/structured"):
        base = root / folder
        if base.exists() and not any(base.iterdir()):
            # Keep the leaf folders present for Obsidian navigation.
            continue
    return removed


def export_obsidian(
    db_path: Path, output_path: Path, timeline_page_size: int = TIMELINE_PAGE_SIZE
) -> dict[str, int]:
    messages = load_messages(db_path)
    structured = load_structured(db_path)
    message_lookup = {message.message_id: message for message in messages}
    messages_by_id = sorted(messages, key=lambda message: message.message_id)
    active_messages = [message for message in messages_by_id if not message.tombstoned]
    expected: set[Path] = set()
    written = 0

    for folder in ("active/messages", "active/structured", "forgotten/messages", "forgotten/structured", "maps"):
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    for index, message in enumerate(messages_by_id):
        folder = "forgotten" if message.tombstoned else "active"
        path = output_path / folder / "messages" / f"{message_stem(message.message_id)}.md"
        previous_message = messages_by_id[index - 1] if index > 0 else None
        next_message = messages_by_id[index + 1] if index + 1 < len(messages_by_id) else None
        expected.add(path)
        if write_if_changed(path, message_markdown(message, structured, previous_message, next_message)):
            written += 1

    for obj in structured:
        message = message_lookup.get(obj.message_id)
        folder = "forgotten" if message is None or message.tombstoned else "active"
        path = output_path / folder / "structured" / f"{structured_stem(obj.id)}.md"
        expected.add(path)
        if write_if_changed(path, structured_markdown(obj, message_lookup)):
            written += 1

    index_path = output_path / "index.md"
    expected.add(index_path)
    if write_if_changed(index_path, index_markdown(messages, structured, timeline_page_size)):
        written += 1

    page_size = max(timeline_page_size, 1)
    page_count = max(1, (len(active_messages) + page_size - 1) // page_size)
    for page_index in range(page_count):
        path = timeline_path(output_path, page_index, page_size)
        expected.add(path)
        if write_if_changed(path, timeline_markdown(active_messages, page_index, page_size)):
            written += 1

    turns_path = output_path / "maps" / "turns.md"
    expected.add(turns_path)
    if write_if_changed(turns_path, turns_markdown(active_messages)):
        written += 1

    removed = clean_stale_markdown(output_path, expected)
    return {
        "messages": len(messages),
        "structured": len(structured),
        "written": written,
        "removed": removed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export RagMemory DB to an Obsidian Markdown mirror.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--timeline-page-size", type=int, default=TIMELINE_PAGE_SIZE)
    parser.add_argument("--clean", action="store_true", help="Remove the output folder before exporting.")
    args = parser.parse_args(argv)

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    stats = export_obsidian(args.db_path, args.output, timeline_page_size=args.timeline_page_size)
    print(
        f"Exported {stats['messages']} message(s), {stats['structured']} structured object(s). "
        f"Written: {stats['written']} | Removed stale: {stats['removed']} | Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
