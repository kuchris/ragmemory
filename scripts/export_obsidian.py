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
OBSIDIAN_GRAPH_COLOR_GROUPS = [
    ("path:\"active/messages\"", 0x00C8F0),
    ("path:\"active/structured\"", 0x009688),
    ("path:\"topic_groups\"", 0x6A5ACD),
    ("path:\"topics\"", 0x9966E6),
    ("path:\"wiki\"", 0x2F80ED),
    ("path:\"files\"", 0xF36C00),
    ("path:\"profile\"", 0xE44DAD),
    ("path:\"forgotten\"", 0xB73636),
]
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
    mode: str
    min_count: int
    allowlist: set[str]
    denylist: set[str]

    def allows(self, tag: str, counts: dict[str, int]) -> bool:
        canonical = canonical_tag(tag)
        if not canonical or canonical in self.denylist:
            return False
        if canonical in self.allowlist:
            return True
        if self.mode == "allowlist":
            return False
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
        mode="count",
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
    if section.get("mode"):
        mode = section["mode"].strip().lower()
        if mode in {"allowlist", "count"}:
            policy.mode = mode
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


def allowed_topic_ids(
    structured: list[StructuredRow],
    topic_policy: TopicPolicy,
    counts: dict[str, int],
) -> set[str]:
    allowed = set()
    for obj in structured:
        for tag in obj.tags:
            topic_id = canonical_tag(tag)
            if topic_id and topic_policy.allows(tag, counts):
                allowed.add(topic_id)
    return allowed


@dataclass(frozen=True)
class Hub:
    stem: str
    hub_type: str
    canonical: str
    title: str | None = None
    description: str | None = None
    aliases: tuple[str, ...] = ()


class HubRegistry:
    def __init__(self):
        self._by_key: dict[tuple[str, str], Hub] = {}
        self._slug_sources: dict[tuple[str, str], str] = {}

    def add(
        self,
        hub_type: str,
        canonical: str,
        title: str | None = None,
        description: str | None = None,
        aliases: list[str] | tuple[str, ...] | None = None,
    ) -> Hub:
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
        hub = Hub(
            stem=stem,
            hub_type=hub_type,
            canonical=canonical,
            title=title,
            description=description,
            aliases=tuple(aliases or ()),
        )
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


def load_topic_taxonomy(db_path: Path) -> dict | None:
    path = db_path / "topic_taxonomy.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "topics" in data and not isinstance(data.get("topics"), list):
        return None
    if "groups" in data and not isinstance(data.get("groups"), list):
        return None
    return data


def build_taxonomy_topic_hubs(
    taxonomy: dict | None,
    registry: HubRegistry,
    valid_group_topic_ids: set[str] | None = None,
) -> tuple[dict[str, list[Hub]], dict[str, list[Hub]], dict[str, list[Hub]]] | None:
    if taxonomy is None:
        return None
    by_structured_id: dict[str, list[Hub]] = {}
    groups_by_topic_id: dict[str, list[Hub]] = {}
    topics_by_group_stem: dict[str, list[Hub]] = {}
    for topic in taxonomy.get("topics", []):
        if not isinstance(topic, dict):
            continue
        topic_id = str(topic.get("id", "")).strip()
        title = str(topic.get("title", "")).strip()
        if not topic_id or not title:
            continue
        aliases = topic.get("aliases", [])
        structured_ids = topic.get("structured_ids", [])
        if not isinstance(aliases, list):
            aliases = []
        if not isinstance(structured_ids, list):
            structured_ids = []
        topic_key = canonical_tag(topic_id)
        if not topic_key:
            continue
        hub = registry.add(
            "topic",
            topic_key,
            title=title,
            description=str(topic.get("description", "")).strip(),
            aliases=[str(item) for item in aliases],
        )
        for structured_id in structured_ids:
            by_structured_id.setdefault(str(structured_id), []).append(hub)
    for group in taxonomy.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id", "")).strip()
        title = str(group.get("title", "")).strip()
        if not group_id or not title:
            continue
        aliases = group.get("aliases", [])
        topic_ids = group.get("topic_ids", group.get("topics", []))
        if not isinstance(aliases, list):
            aliases = []
        if not isinstance(topic_ids, list):
            topic_ids = []
        group_hub = registry.add(
            "topic_group",
            group_id,
            title=title,
            description=str(group.get("description", "")).strip(),
            aliases=[str(item) for item in aliases],
        )
        for topic_id in topic_ids:
            topic_key = canonical_tag(str(topic_id))
            if not topic_key:
                continue
            if valid_group_topic_ids is not None and topic_key not in valid_group_topic_ids:
                continue
            topic_hub = registry.add("topic", topic_key)
            groups_by_topic_id.setdefault(topic_key, []).append(group_hub)
            topics_by_group_stem.setdefault(group_hub.stem, []).append(topic_hub)
    return by_structured_id, groups_by_topic_id, topics_by_group_stem


def message_preview(message: MessageRow, limit: int = 120) -> str:
    return escape_raw_wikilinks(" ".join(message_display_text(message).split())[:limit])


def hubs_for_structured(
    obj: StructuredRow,
    registry: HubRegistry,
    topic_policy: TopicPolicy,
    counts: dict[str, int],
    file_policy: FileHubPolicy,
    taxonomy_topic_hubs: dict[str, list[Hub]] | None = None,
    taxonomy_group_hubs: dict[str, list[Hub]] | None = None,
    create: bool = False,
) -> list[Hub]:
    hubs: list[Hub] = []
    if taxonomy_topic_hubs is not None:
        hubs.extend(taxonomy_topic_hubs.get(obj.id, []))
    seen_tags = set()
    for tag in sorted(obj.tags, key=lambda value: value.lower()):
        key = canonical_tag(tag)
        seen_key = tag.strip().lower()
        if not key or not seen_key or seen_key in seen_tags:
            continue
        seen_tags.add(seen_key)
        if not topic_policy.allows(tag, counts):
            continue
        hub = registry.add("topic", key, title=tag.strip() or None) if create else registry.get("topic", key)
        if hub:
            hubs.append(hub)
            if taxonomy_group_hubs is not None:
                hubs.extend(taxonomy_group_hubs.get(key, []))

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
    taxonomy_topic_hubs: dict[str, list[Hub]] | None = None,
    taxonomy_group_hubs: dict[str, list[Hub]] | None = None,
    registry: HubRegistry | None = None,
) -> HubRegistry:
    registry = registry or HubRegistry()
    for obj in structured:
        hubs_for_structured(
            obj,
            registry,
            topic_policy,
            counts,
            file_policy,
            taxonomy_topic_hubs=taxonomy_topic_hubs,
            taxonomy_group_hubs=taxonomy_group_hubs,
            create=True,
        )
    return registry


def hub_markdown(hub: Hub, topic_group_members: dict[str, list[Hub]] | None = None) -> str:
    title = hub.title or hub.stem
    parts = [
        frontmatter(
            {
                "generated": "ragmemory-export",
                "hub_type": hub.hub_type,
                "canonical": hub.canonical,
                "aliases": list(hub.aliases),
                "cssclasses": ["hub"],
            }
        ),
        "",
        f"# {title}",
        "",
        hub.description or f"Auto-generated `{hub.hub_type}` hub for `{hub.canonical}`.",
        "",
    ]
    if topic_group_members and hub.hub_type == "topic_group":
        members = topic_group_members.get(hub.stem, [])
        if members:
            parts.extend(["## Topics", ""])
            for member in members:
                parts.append(f"- {wikilink(member.stem)}")
            parts.append("")
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
    taxonomy_topic_hubs: dict[str, list[Hub]] | None = None,
    taxonomy_group_hubs: dict[str, list[Hub]] | None = None,
) -> str:
    message = message_lookup.get(obj.message_id)
    status = "forgotten" if message and message.tombstoned else "active"
    hubs = hubs_for_structured(
        obj,
        hub_registry,
        topic_policy,
        counts,
        file_policy,
        taxonomy_topic_hubs=taxonomy_topic_hubs,
        taxonomy_group_hubs=taxonomy_group_hubs,
        create=False,
    )
    topic_hubs = [hub for hub in hubs if hub.hub_type == "topic"]
    topic_group_hubs = [hub for hub in hubs if hub.hub_type == "topic_group"]
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
    if topic_group_hubs:
        parts.append("- Topic Groups: " + ", ".join(wikilink(hub.stem) for hub in topic_group_hubs))
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


def is_stale_root_structured_note(path: Path) -> bool:
    if not path.exists() or not path.name.startswith("sm_") or path.suffix != ".md":
        return False
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.strip():
        return True
    if not text.startswith("---\n") or text.count("---") < 2:
        return False
    frontmatter_text = text.split("---", 2)[1]
    return (
        "generated: \"ragmemory-export\"\n" in frontmatter_text
        and "cssclasses: [\"memory-structured\"]" in frontmatter_text
    )


def is_blank_obsidian_untitled(path: Path) -> bool:
    if path.name == "Untitled.canvas":
        return path.read_text(encoding="utf-8").strip() == "{}"
    if path.name == "Untitled.base":
        return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip() == (
            "views:\n  - type: table\n    name: Table"
        )
    return False


def configure_obsidian_graph(output_path: Path) -> Path:
    graph_path = output_path / ".obsidian" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    if graph_path.exists():
        raw_graph = graph_path.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(raw_graph)
        except json.JSONDecodeError:
            try:
                data, _ = json.JSONDecoder().raw_decode(raw_graph)
            except json.JSONDecodeError:
                data = {}
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    data.setdefault("collapse-filter", True)
    data.setdefault("search", "")
    data.setdefault("showTags", False)
    data.setdefault("showAttachments", False)
    data.setdefault("hideUnresolved", False)
    data.setdefault("showOrphans", True)
    data["collapse-color-groups"] = False
    data["colorGroups"] = [
        {"query": query, "color": {"a": 1, "rgb": rgb}}
        for query, rgb in OBSIDIAN_GRAPH_COLOR_GROUPS
    ]
    graph_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return graph_path


def clean_stale_markdown(root: Path, expected: set[Path]) -> int:
    removed = 0
    for name in ("Untitled.base", "Untitled.canvas"):
        path = root / name
        if path.exists() and is_blank_obsidian_untitled(path):
            path.unlink()
            removed += 1
    for path in root.glob("sm_*.md"):
        if path not in expected and is_stale_root_structured_note(path):
            path.unlink()
            removed += 1
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
    for folder in ("topic_groups", "topics", "files", "profile"):
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
    taxonomy = load_topic_taxonomy(db_path)
    active_structured = active_structured_rows(structured, message_lookup)
    counts = topic_counts(active_structured)
    valid_group_topic_ids = allowed_topic_ids(active_structured, topic_policy, counts)
    hub_registry = HubRegistry()
    taxonomy_hubs = build_taxonomy_topic_hubs(taxonomy, hub_registry, valid_group_topic_ids)
    taxonomy_topic_hubs: dict[str, list[Hub]] | None = None
    taxonomy_group_hubs: dict[str, list[Hub]] | None = None
    topic_group_members: dict[str, list[Hub]] | None = None
    if taxonomy_hubs is not None:
        taxonomy_topic_hubs, taxonomy_group_hubs, topic_group_members = taxonomy_hubs
    hub_registry = build_hub_registry(
        active_structured,
        topic_policy,
        counts,
        file_policy,
        taxonomy_topic_hubs=taxonomy_topic_hubs,
        taxonomy_group_hubs=taxonomy_group_hubs,
        registry=hub_registry,
    )
    messages_by_id = sorted(messages, key=lambda message: message.message_id)
    active_messages = [message for message in messages_by_id if not message.tombstoned]
    expected: set[Path] = set()
    written = 0

    for folder in (
        "active/messages",
        "active/structured",
        "forgotten/messages",
        "forgotten/structured",
        "maps",
        "topic_groups",
        "topics",
        "files",
        "profile",
    ):
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    for hub in hub_registry.values():
        path = output_path / f"{hub.stem}.md"
        expected.add(path)
        if write_if_changed(path, hub_markdown(hub, topic_group_members)):
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
        if write_if_changed(
            path,
            structured_markdown(
                obj,
                message_lookup,
                hub_registry,
                topic_policy,
                counts,
                file_policy,
                taxonomy_topic_hubs=taxonomy_topic_hubs,
                taxonomy_group_hubs=taxonomy_group_hubs,
            ),
        ):
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
    configure_obsidian_graph(output_path)
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
