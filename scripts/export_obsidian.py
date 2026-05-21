"""
Export RagMemory state to an Obsidian-readable Markdown mirror.

The mirror is a generated one-way projection. Edit the DB through RagMemory,
then re-run this script; do not treat the Markdown files as source of truth.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("./.data/chroma_db")
DEFAULT_OUTPUT_PATH = Path("./.data/obsidian_memory")
TIMELINE_PAGE_SIZE = 500
PATH_RE = re.compile(
    r"\b(?:[\w.-]+/){1,8}[\w.-]+\."
    r"(?:py|ts|js|rs|go|md|json|yaml|yml|toml|sql|sh|css|html|tsx|jsx)\b"
)
EVIDENCE_REF_RE = re.compile(r"\bevidence\[([a-z]+):([a-f0-9]{12})\]")
FILE_HUB_TYPES = {"code_reference", "config"}
PROFILE_HUB_TYPES = {"preference", "identity"}
DEFAULT_TOPIC_MIN_COUNT = 2
DEFAULT_TOPIC_ALLOWLIST = {"ragmemory", "obsidian", "codex-hooks", "memory-decay"}
DEFAULT_TOPIC_DENYLIST = {
    "code_reference",
    "config",
    "decision",
    "constraint",
    "preference",
    "open_question",
    "chart",
    "table",
    "text",
    "profile",
    "python",
    "powershell",
    "javascript",
    "typescript",
    "bash",
    "ini",
    "yaml",
    "json",
    "markdown",
    "sql",
    "html",
    "css",
}


@dataclass
class MessageRow:
    message_id: int
    role: str
    text: str
    content_hash: str
    created_at: str | None
    tombstoned: bool
    compact_text: str | None = None
    compact_status: str | None = None


def message_display_text(message: MessageRow) -> str:
    if message.compact_status == "ok" and message.compact_text and message.compact_text.strip():
        return message.compact_text
    return message.text


def message_text_source(message: MessageRow) -> str:
    if message.compact_status == "ok" and message.compact_text and message.compact_text.strip():
        return "compact_text"
    return "raw_text"


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
    content_hash: str | None = None


@dataclass
class TopicPolicy:
    min_count: int
    allowlist: set[str]
    denylist: set[str]

    def allows(self, tag: str, counts: dict[str, int]) -> bool:
        canonical = canonical_tag(tag)
        if not canonical or canonical in self.denylist:
            return False
        if canonical in self.allowlist:
            return True
        return counts.get(canonical, 0) >= self.min_count


@dataclass
class FileHubPolicy:
    enabled: bool = False


def split_config_list(value: str) -> set[str]:
    return {
        canonical_tag(item)
        for item in re.split(r"[,\n]+", value)
        if canonical_tag(item)
    }


def load_topic_policy(config_path: Path | None = None) -> TopicPolicy:
    policy = TopicPolicy(
        min_count=DEFAULT_TOPIC_MIN_COUNT,
        allowlist=set(DEFAULT_TOPIC_ALLOWLIST),
        denylist=set(DEFAULT_TOPIC_DENYLIST),
    )
    if config_path is None:
        config_path = Path.cwd() / "ragmemory.local.ini"
    if not config_path.exists():
        return policy

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8-sig")
    if not parser.has_section("obsidian.topics"):
        return policy

    section = parser["obsidian.topics"]
    if section.get("min_count"):
        try:
            policy.min_count = max(1, int(section["min_count"]))
        except ValueError:
            pass
    if section.get("allowlist"):
        policy.allowlist = split_config_list(section["allowlist"])
    if section.get("denylist"):
        policy.denylist = split_config_list(section["denylist"])
    return policy


def load_file_hub_policy(config_path: Path | None = None) -> FileHubPolicy:
    policy = FileHubPolicy()
    if config_path is None:
        config_path = Path.cwd() / "ragmemory.local.ini"
    if not config_path.exists():
        return policy

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8-sig")
    if not parser.has_section("obsidian.files"):
        return policy
    policy.enabled = parser["obsidian.files"].getboolean("enable", fallback=False)
    return policy


def load_messages(db_path: Path) -> list[MessageRow]:
    state_db = db_path / "state.sqlite"
    if not state_db.exists():
        return []
    with sqlite3.connect(state_db) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        compact_text_expr = "compact_text" if "compact_text" in columns else "NULL"
        compact_status_expr = "compact_status" if "compact_status" in columns else "NULL"
        rows = conn.execute(
            f"""
            SELECT
                message_id, role, text, content_hash, created_at, tombstoned,
                {compact_text_expr}, {compact_status_expr}
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
            compact_text=compact_text,
            compact_status=compact_status,
        )
        for (
            message_id, role, text, content_hash, created_at, tombstoned,
            compact_text, compact_status,
        ) in rows
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
                content_hash=str(item.get("content_hash", "")).strip() or None,
            )
        )
    return rows


def message_stem(message_id: int) -> str:
    return f"msg-{message_id:06d}"


def structured_stem(object_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", object_id).strip("-") or "structured"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "untitled"


def stable_suffix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def canonical_tag(tag: str) -> str:
    return slugify(tag)


def active_structured_rows(
    structured: list[StructuredRow],
    message_lookup: dict[int, MessageRow],
) -> list[StructuredRow]:
    return [
        obj for obj in structured
        if not message_lookup.get(
            obj.message_id,
            MessageRow(obj.message_id, obj.role, "", "", None, True),
        ).tombstoned
    ]


def topic_counts(structured: list[StructuredRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in structured:
        seen = set()
        for tag in obj.tags:
            canonical = canonical_tag(tag)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            counts[canonical] = counts.get(canonical, 0) + 1
    return counts


@dataclass(frozen=True)
class Hub:
    stem: str
    hub_type: str
    canonical: str


class HubRegistry:
    def __init__(self):
        self._by_key: dict[tuple[str, str], Hub] = {}
        self._slug_sources: dict[tuple[str, str], str] = {}

    def add(self, hub_type: str, canonical: str) -> Hub:
        canonical = canonical.strip()
        key = (hub_type, canonical.lower())
        if key in self._by_key:
            return self._by_key[key]

        base_slug = slugify(canonical)
        slug_key = (hub_type, base_slug)
        existing = self._slug_sources.get(slug_key)
        slug = base_slug
        if existing is not None and existing != canonical.lower():
            slug = f"{base_slug}-{stable_suffix(canonical.lower())}"
        else:
            self._slug_sources[slug_key] = canonical.lower()

        stem = "profile/user" if hub_type == "profile" else f"{hub_type}s/{slug}"
        hub = Hub(stem=stem, hub_type=hub_type, canonical=canonical)
        self._by_key[key] = hub
        return hub

    def get(self, hub_type: str, canonical: str) -> Hub | None:
        return self._by_key.get((hub_type, canonical.strip().lower()))

    def values(self) -> list[Hub]:
        return sorted(self._by_key.values(), key=lambda hub: hub.stem)


def file_paths_from_source(text: str) -> list[str]:
    paths = []
    seen = set()
    for match in PATH_RE.findall(text):
        path = match.strip().strip("`'\".,;:()[]{}")
        if len(path) < 4 or len(path) > 128 or " " in path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


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


def escape_raw_wikilinks(text: str) -> str:
    return text.replace("[[", "&#91;&#91;").replace("]]", "&#93;&#93;")


def evidence_ref_type(obj: StructuredRow) -> str:
    for tag in obj.tags:
        canonical = canonical_tag(tag)
        if canonical and canonical != "code-reference":
            return canonical.replace("-", "")
    return canonical_tag(obj.type).replace("-", "") or "text"


def evidence_marker(obj: StructuredRow) -> str | None:
    if not obj.content_hash:
        return None
    return f"evidence[{evidence_ref_type(obj)}:{obj.content_hash}]"


def evidence_lookup(structured: list[StructuredRow]) -> dict[str, list[StructuredRow]]:
    lookup: dict[str, list[StructuredRow]] = {}
    for obj in structured:
        marker = evidence_marker(obj)
        if marker:
            lookup.setdefault(marker, []).append(obj)
    return lookup


def evidence_refs_for_message(message: MessageRow, structured: list[StructuredRow]) -> list[tuple[str, StructuredRow | None]]:
    markers = [
        f"evidence[{ref_type}:{content_hash}]"
        for ref_type, content_hash in EVIDENCE_REF_RE.findall(message_display_text(message))
    ]
    if not markers:
        return []
    lookup = evidence_lookup(structured)
    refs = []
    seen = set()
    for marker in markers:
        if marker in seen:
            continue
        seen.add(marker)
        matches = lookup.get(marker, [])
        obj = next((item for item in matches if item.message_id == message.message_id), None)
        if obj is None and matches:
            obj = matches[0]
        refs.append((marker, obj))
    return refs


def message_preview(message: MessageRow, limit: int = 120) -> str:
    return escape_raw_wikilinks(" ".join(message_display_text(message).split())[:limit])


def hubs_for_structured(
    obj: StructuredRow,
    registry: HubRegistry,
    topic_policy: TopicPolicy,
    counts: dict[str, int],
    file_policy: FileHubPolicy,
    create: bool = False,
) -> list[Hub]:
    hubs: list[Hub] = []
    seen_tags = set()
    for tag in sorted(obj.tags, key=lambda value: value.lower()):
        key = tag.strip().lower()
        if not key or key in seen_tags:
            continue
        seen_tags.add(key)
        if not topic_policy.allows(tag, counts):
            continue
        hub = registry.add("topic", tag) if create else registry.get("topic", tag)
        if hub:
            hubs.append(hub)

    if obj.type in PROFILE_HUB_TYPES:
        hub = registry.add("profile", "user") if create else registry.get("profile", "user")
        if hub:
            hubs.append(hub)

    if file_policy.enabled and obj.type in FILE_HUB_TYPES:
        for path in file_paths_from_source(obj.source_text):
            hub = registry.add("file", path) if create else registry.get("file", path)
            if hub:
                hubs.append(hub)

    deduped = []
    seen_stems = set()
    for hub in hubs:
        if hub.stem not in seen_stems:
            seen_stems.add(hub.stem)
            deduped.append(hub)
    return deduped


def build_hub_registry(
    structured: list[StructuredRow],
    topic_policy: TopicPolicy,
    counts: dict[str, int],
    file_policy: FileHubPolicy,
) -> HubRegistry:
    registry = HubRegistry()
    for obj in structured:
        hubs_for_structured(obj, registry, topic_policy, counts, file_policy, create=True)
    return registry


def hub_markdown(hub: Hub) -> str:
    title = hub.stem
    parts = [
        frontmatter(
            {
                "generated": "ragmemory-export",
                "hub_type": hub.hub_type,
                "canonical": hub.canonical,
                "cssclasses": ["hub"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"Auto-generated `{hub.hub_type}` hub for `{hub.canonical}`.",
        "",
    ]
    return "\n".join(parts)


def message_markdown(
    message: MessageRow,
    structured: list[StructuredRow],
    previous_message: MessageRow | None = None,
    next_message: MessageRow | None = None,
) -> str:
    related = [obj for obj in structured if obj.message_id == message.message_id]
    cssclasses = ["memory-message"]
    if not related:
        cssclasses.append("memory-unlinked")
    parts = [
        frontmatter(
            {
                "message_id": message.message_id,
                "role": message.role,
                "status": "forgotten" if message.tombstoned else "active",
                "created_at": message.created_at,
                "content_hash": message.content_hash,
                "text_source": message_text_source(message),
                "compact_status": message.compact_status,
                "structured_objects": [structured_stem(obj.id) for obj in related],
                "previous_message": message_stem(previous_message.message_id) if previous_message else None,
                "next_message": message_stem(next_message.message_id) if next_message else None,
                "cssclasses": cssclasses,
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
    parts.append(f"- Text source: `{message_text_source(message)}`")
    if related:
        parts.append("- Structured: " + ", ".join(wikilink(structured_stem(obj.id)) for obj in related))
    parts.extend(["", "## Text", "", escape_raw_wikilinks(message_display_text(message).strip()), ""])
    evidence_refs = evidence_refs_for_message(message, structured)
    if evidence_refs:
        parts.extend(["## Evidence References", ""])
        for marker, obj in evidence_refs:
            if obj:
                parts.append(f"- `{marker}` -> {wikilink(structured_stem(obj.id))}")
            else:
                parts.append(f"- `{marker}` -> unresolved")
        parts.append("")
    return "\n".join(parts)


def structured_markdown(
    obj: StructuredRow,
    message_lookup: dict[int, MessageRow],
    hub_registry: HubRegistry,
    topic_policy: TopicPolicy,
    counts: dict[str, int],
    file_policy: FileHubPolicy,
) -> str:
    message = message_lookup.get(obj.message_id)
    status = "forgotten" if message and message.tombstoned else "active"
    hubs = hubs_for_structured(obj, hub_registry, topic_policy, counts, file_policy, create=False)
    topic_hubs = [hub for hub in hubs if hub.hub_type == "topic"]
    file_hubs = [hub for hub in hubs if hub.hub_type == "file"]
    profile_hubs = [hub for hub in hubs if hub.hub_type == "profile"]
    marker = evidence_marker(obj)
    parts = [
        frontmatter(
            {
                "structured_id": obj.id,
                "type": obj.type,
                "status": status,
                "message_id": obj.message_id,
                "role": obj.role,
                "importance": obj.importance,
                "content_hash": obj.content_hash,
                "evidence_ref": marker,
                "cssclasses": ["memory-structured"],
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
    if marker:
        parts.append(f"- Evidence ref: `{marker}`")
    if topic_hubs:
        parts.append("- Topics: " + ", ".join(wikilink(hub.stem) for hub in topic_hubs))
    if file_hubs:
        parts.append("- Files: " + ", ".join(wikilink(hub.stem) for hub in file_hubs))
    if profile_hubs:
        parts.append("- Profile: " + ", ".join(wikilink(hub.stem) for hub in profile_hubs))
    parts.extend([
        "",
        "## Summary",
        "",
        escape_raw_wikilinks(obj.summary.strip()),
        "",
        "## Source Text",
        "",
        escape_raw_wikilinks(obj.source_text.strip()),
        "",
    ])
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
                "cssclasses": ["navigation"],
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
            preview = message_preview(message)
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
        "- active/messages",
        "- active/structured",
        "- forgotten/messages",
        "- forgotten/structured",
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
                "cssclasses": ["navigation"],
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
            preview = message_preview(message)
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
                "cssclasses": ["navigation"],
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
            preview = message_preview(message)
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


def write_if_missing(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text(content.replace("\r\n", "\n"), encoding="utf-8")
    return True


def is_generated_hub(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.strip():
        return True
    return (
        text.startswith("---\n")
        and (
            "generated: \"ragmemory-export\"\n" in text.split("---", 2)[1]
            or "generated: true\n" in text.split("---", 2)[1]
        )
        and "cssclasses: [\"hub\"]" in text.split("---", 2)[1]
    )


def clean_stale_markdown(root: Path, expected: set[Path]) -> int:
    removed = 0
    generated_folders = (
        "active/messages",
        "active/structured",
        "forgotten/messages",
        "forgotten/structured",
        "maps",
    )
    for folder in generated_folders:
        base = root / folder
        if not base.exists():
            continue
        for path in base.glob("*.md"):
            if path not in expected:
                path.unlink()
                removed += 1
    for folder in ("topics", "files", "profile"):
        base = root / folder
        if not base.exists():
            continue
        for path in base.glob("*.md"):
            if path not in expected and is_generated_hub(path):
                path.unlink()
                removed += 1
    for folder in ("active/messages", "active/structured", "forgotten/messages", "forgotten/structured"):
        base = root / folder
        if base.exists() and not any(base.iterdir()):
            # Keep the leaf folders present for Obsidian navigation.
            continue
    return removed


def export_obsidian(
    db_path: Path,
    output_path: Path,
    timeline_page_size: int = TIMELINE_PAGE_SIZE,
    config_path: Path | None = None,
) -> dict[str, int]:
    messages = load_messages(db_path)
    structured = load_structured(db_path)
    message_lookup = {message.message_id: message for message in messages}
    topic_policy = load_topic_policy(config_path)
    file_policy = load_file_hub_policy(config_path)
    counts = topic_counts(active_structured_rows(structured, message_lookup))
    hub_registry = build_hub_registry(
        active_structured_rows(structured, message_lookup),
        topic_policy,
        counts,
        file_policy,
    )
    messages_by_id = sorted(messages, key=lambda message: message.message_id)
    active_messages = [message for message in messages_by_id if not message.tombstoned]
    expected: set[Path] = set()
    written = 0

    for folder in ("active/messages", "active/structured", "forgotten/messages", "forgotten/structured", "maps", "topics", "files", "profile"):
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    for hub in hub_registry.values():
        path = output_path / f"{hub.stem}.md"
        expected.add(path)
        if write_if_missing(path, hub_markdown(hub)):
            written += 1

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
        if write_if_changed(path, structured_markdown(obj, message_lookup, hub_registry, topic_policy, counts, file_policy)):
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
    parser.add_argument("--config", type=Path, help="Optional ragmemory local settings file.")
    parser.add_argument("--clean", action="store_true", help="Remove the output folder before exporting.")
    args = parser.parse_args(argv)

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    stats = export_obsidian(
        args.db_path,
        args.output,
        timeline_page_size=args.timeline_page_size,
        config_path=args.config,
    )
    print(
        f"Exported {stats['messages']} message(s), {stats['structured']} structured object(s). "
        f"Written: {stats['written']} | Removed stale: {stats['removed']} | Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
