"""
Generate a non-LLM wiki layer from the Obsidian memory graph.

Run:
    uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ragmemory.llm import DEFAULT_LLM_PROVIDER, LLMProviderClient, LLMProviderOptions, provider_env_prefix


DEFAULT_OBSIDIAN_PATH = Path("./.data/obsidian_memory")
DEFAULT_CONFIG_PATH = Path("./ragmemory.local.ini")
DEFAULT_WIKI_MAX_TOKENS = 1200
WIKILINK_RE = re.compile(r"\[\[([^|\]#]+)")


@dataclass
class TopicGroup:
    stem: str
    title: str
    description: str
    topics: list[str]


@dataclass
class StructuredNote:
    stem: str
    title: str
    memory_type: str
    status: str
    message_id: int
    importance: float
    summary: str
    topics: list[str]
    topic_groups: list[str]


@dataclass
class WikiLLMOptions:
    enabled: bool = False
    provider: str = DEFAULT_LLM_PROVIDER
    model: str = ""
    max_tokens: int = DEFAULT_WIKI_MAX_TOKENS
    thinking: str = ""
    force: bool = False
    limit: int | None = None


def wikilink(stem: str) -> str:
    return f"[[{stem}]]"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def load_config(path: Path) -> None:
    if not path.exists():
        return
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8-sig")
    if parser.has_section("llm"):
        section = parser["llm"]
        if section.get("structured_provider"):
            os.environ.setdefault("RAGMEMORY_STRUCTURED_PROVIDER", section["structured_provider"].strip())
    if parser.has_section("topic_regroup"):
        section = parser["topic_regroup"]
        for key, env_name in (
            ("provider", "RAGMEMORY_TOPIC_PROVIDER"),
            ("model", "RAGMEMORY_TOPIC_MODEL"),
            ("thinking", "RAGMEMORY_TOPIC_THINKING"),
        ):
            if section.get(key):
                os.environ.setdefault(env_name, section[key].strip())
    if parser.has_section("wiki"):
        section = parser["wiki"]
        for key, env_name in (
            ("provider", "RAGMEMORY_WIKI_PROVIDER"),
            ("model", "RAGMEMORY_WIKI_MODEL"),
            ("max_tokens", "RAGMEMORY_WIKI_MAX_TOKENS"),
            ("thinking", "RAGMEMORY_WIKI_THINKING"),
        ):
            if section.get(key):
                os.environ.setdefault(env_name, section[key].strip())
    for section_name in parser.sections():
        if not section_name.startswith("llm."):
            continue
        provider = section_name.split(".", 1)[1].strip()
        if not provider:
            continue
        prefix = provider_env_prefix(provider)
        section = parser[section_name]
        for key, env_suffix in (
            ("api_key", "API_KEY"),
            ("base_url", "BASE_URL"),
            ("model", "MODEL"),
            ("api_style", "API_STYLE"),
            ("thinking", "THINKING"),
        ):
            if section.get(key):
                os.environ.setdefault(f"{prefix}_{env_suffix}", section[key].strip())
        if provider.lower() == "nvidia" and section.get("api_key"):
            os.environ.setdefault("NVIDIA_API_KEY", section["api_key"].strip())


def wiki_llm_options_from_env(enabled: bool, force: bool, limit: int | None) -> WikiLLMOptions:
    provider = os.environ.get(
        "RAGMEMORY_WIKI_PROVIDER",
        os.environ.get(
            "RAGMEMORY_TOPIC_PROVIDER",
            os.environ.get("RAGMEMORY_STRUCTURED_PROVIDER", DEFAULT_LLM_PROVIDER),
        ),
    ).strip().lower()
    model = os.environ.get(
        "RAGMEMORY_WIKI_MODEL",
        os.environ.get("RAGMEMORY_TOPIC_MODEL", ""),
    ).strip()
    return WikiLLMOptions(
        enabled=enabled or env_bool("RAGMEMORY_WIKI_ENABLE", False),
        provider=provider,
        model=model,
        max_tokens=max(1, env_int("RAGMEMORY_WIKI_MAX_TOKENS", DEFAULT_WIKI_MAX_TOKENS)),
        thinking=os.environ.get(
            "RAGMEMORY_WIKI_THINKING",
            os.environ.get("RAGMEMORY_TOPIC_THINKING", ""),
        ).strip().lower(),
        force=force,
        limit=limit,
    )


def frontmatter(fields: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            rendered = json.dumps(value, ensure_ascii=False)
        elif value is None:
            rendered = "null"
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    fields: dict[str, object] = {}
    for raw_line in parts[1].splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            fields[key] = json.loads(value)
        except json.JSONDecodeError:
            fields[key] = value.strip('"')
    return fields, parts[2]


def first_heading(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def first_paragraph_after_heading(body: str) -> str:
    lines = body.splitlines()
    seen_heading = False
    paragraph = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            seen_heading = True
            continue
        if not seen_heading:
            continue
        if stripped.startswith("## "):
            break
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


def section_text(body: str, heading: str) -> str:
    marker = f"## {heading}"
    lines = body.splitlines()
    capture = False
    captured = []
    for line in lines:
        if line.strip() == marker:
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture:
            captured.append(line)
    return "\n".join(captured).strip()


def body_links(body: str, prefix: str | None = None) -> list[str]:
    links = []
    seen = set()
    for target in WIKILINK_RE.findall(body):
        target = target.strip()
        if prefix and not target.startswith(prefix):
            continue
        if target and target not in seen:
            seen.add(target)
            links.append(target)
    return links


def load_topic_groups(root: Path) -> list[TopicGroup]:
    groups = []
    for path in sorted((root / "topic_groups").glob("*.md")):
        fields, body = split_frontmatter(path.read_text(encoding="utf-8-sig"))
        stem = path.relative_to(root).with_suffix("").as_posix()
        groups.append(
            TopicGroup(
                stem=stem,
                title=first_heading(body, str(fields.get("canonical") or path.stem)),
                description=first_paragraph_after_heading(body),
                topics=body_links(body, "topics/"),
            )
        )
    return groups


def load_structured_notes(root: Path, include_forgotten: bool = False) -> list[StructuredNote]:
    folders = [root / "active" / "structured"]
    if include_forgotten:
        folders.append(root / "forgotten" / "structured")
    notes = []
    for folder in folders:
        for path in sorted(folder.glob("*.md")):
            fields, body = split_frontmatter(path.read_text(encoding="utf-8-sig"))
            stem = path.relative_to(root).with_suffix("").as_posix()
            message_id = fields.get("message_id") or 0
            importance = fields.get("importance") or 0.0
            notes.append(
                StructuredNote(
                    stem=stem,
                    title=first_heading(body, path.stem),
                    memory_type=str(fields.get("type") or "memory"),
                    status=str(fields.get("status") or "active"),
                    message_id=int(message_id) if isinstance(message_id, int | float | str) and str(message_id).isdigit() else 0,
                    importance=float(importance) if isinstance(importance, int | float | str) else 0.0,
                    summary=" ".join(section_text(body, "Summary").split()),
                    topics=body_links(body, "topics/"),
                    topic_groups=body_links(body, "topic_groups/"),
                )
            )
    return notes


def sort_notes(notes: list[StructuredNote]) -> list[StructuredNote]:
    return sorted(notes, key=lambda note: (-note.importance, -note.message_id, note.stem))


def note_bullet(note: StructuredNote) -> str:
    summary = note.summary or note.title
    return f"- {wikilink(note.stem)} `{note.memory_type}` msg `{note.message_id}`: {summary}"


def render_note_section(title: str, notes: list[StructuredNote], max_items: int) -> list[str]:
    if not notes:
        return []
    rendered = [f"## {title}", ""]
    for note in sort_notes(notes)[:max_items]:
        rendered.append(note_bullet(note))
    if len(notes) > max_items:
        rendered.append(f"- ... {len(notes) - max_items} more")
    rendered.append("")
    return rendered


def compact_note_payload(notes: list[StructuredNote], limit: int) -> list[dict[str, object]]:
    payload = []
    for note in sort_notes(notes)[:limit]:
        payload.append(
            {
                "id": note.stem,
                "type": note.memory_type,
                "message_id": note.message_id,
                "importance": note.importance,
                "summary": note.summary or note.title,
                "topics": note.topics[:8],
            }
        )
    return payload


def build_llm_payload(group: TopicGroup, notes: list[StructuredNote], max_notes: int) -> dict[str, object]:
    return {
        "group": {
            "id": group.stem,
            "title": group.title,
            "description": group.description,
            "topics": group.topics,
        },
        "structured_memories": compact_note_payload(notes, max_notes),
    }


def llm_cache_key(group: TopicGroup, notes: list[StructuredNote], max_notes: int, model: str) -> str:
    payload = build_llm_payload(group, notes, max_notes)
    payload["model"] = model
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_llm_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def save_llm_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_wiki_llm_prompt(group: TopicGroup, notes: list[StructuredNote], max_notes: int) -> str:
    return f"""Write a concise wiki summary for one RagMemory topic group.

Use only the provided JSON. Do not invent facts.
Keep the answer short and useful for a developer inspecting the Obsidian graph.
Return Markdown only with these headings:

## LLM Summary
## What To Look At
## Useful Next Questions

Rules:
- Mention uncertainty if the memories are thin or noisy.
- Prefer concrete project/module names over generic wording.
- Do not include frontmatter.
- Do not include code fences.
- Use ASCII punctuation only.

Input:
{json.dumps(build_llm_payload(group, notes, max_notes), ensure_ascii=False)}
"""


def ascii_clean(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2022": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", errors="replace").decode("ascii")


def build_llm_client(options: WikiLLMOptions) -> LLMProviderClient:
    llm_options = LLMProviderOptions.from_env(options.provider, options.model)
    if options.thinking:
        llm_options.extra_body = LLMProviderOptions._extra_body_for(llm_options.provider, options.thinking)
    return LLMProviderClient(llm_options)


def generate_llm_summary(
    group: TopicGroup,
    notes: list[StructuredNote],
    client: LLMProviderClient,
    options: WikiLLMOptions,
    max_notes: int,
) -> str | None:
    content = client.complete_chat(
        messages=[
            {
                "role": "system",
                "content": "You write concise local wiki pages from RagMemory graph evidence. Return Markdown only.",
            },
            {"role": "user", "content": build_wiki_llm_prompt(group, notes, max_notes)},
        ],
        temperature=0,
        max_tokens=options.max_tokens,
    )
    if content is None:
        print(f"  [wiki llm skipped] {group.stem}: {client.last_error}")
        return None
    return ascii_clean(content.strip())


def page_name_for_group(group: TopicGroup) -> str:
    return group.stem.split("/", 1)[1]


def render_group_page(
    group: TopicGroup,
    notes: list[StructuredNote],
    max_topics: int,
    max_items_per_section: int,
    llm_summary: str | None = None,
) -> str:
    by_type: dict[str, list[StructuredNote]] = {}
    for note in notes:
        by_type.setdefault(note.memory_type, []).append(note)
    parts = [
        frontmatter(
            {
                "generated": "ragmemory-wiki",
                "source_group": group.stem,
                "structured_count": len(notes),
                "topic_count": len(group.topics),
                "cssclasses": ["wiki"],
            }
        ),
        "",
        f"# {group.title}",
        "",
        "Generated wiki page from the RagMemory Obsidian graph. Edit source memory or taxonomy, then regenerate.",
        "",
        "## Overview",
        "",
        f"- Source group: {wikilink(group.stem)}",
        f"- Related topics: `{len(group.topics)}`",
        f"- Connected structured memories: `{len(notes)}`",
    ]
    if group.description:
        parts.append(f"- Group description: {group.description}")
    if llm_summary:
        parts.extend(["", llm_summary.strip(), ""])
    parts.extend(["", "## Related Topics", ""])
    for topic in group.topics[:max_topics]:
        parts.append(f"- {wikilink(topic)}")
    if len(group.topics) > max_topics:
        parts.append(f"- ... {len(group.topics) - max_topics} more")
    parts.append("")

    parts.extend(render_note_section("Key Decisions", by_type.get("decision", []), max_items_per_section))
    parts.extend(render_note_section("Preferences And Constraints", by_type.get("preference", []) + by_type.get("constraint", []), max_items_per_section))
    parts.extend(render_note_section("Configs And Code References", by_type.get("config", []) + by_type.get("code_reference", []), max_items_per_section))
    parts.extend(render_note_section("Open Questions", by_type.get("open_question", []), max_items_per_section))
    used_types = {"decision", "preference", "constraint", "config", "code_reference", "open_question"}
    other_notes = [note for note in notes if note.memory_type not in used_types]
    parts.extend(render_note_section("Other Memory", other_notes, max_items_per_section))
    return "\n".join(parts).rstrip() + "\n"


def render_index(groups: list[TopicGroup], group_notes: dict[str, list[StructuredNote]]) -> str:
    parts = [
        frontmatter({"generated": "ragmemory-wiki", "cssclasses": ["wiki", "navigation"]}),
        "",
        "# RagMemory Wiki",
        "",
        "Generated index for topic-group wiki pages. Edit source memory or taxonomy, then regenerate.",
        "",
        "## Topic Groups",
        "",
    ]
    for group in sorted(groups, key=lambda item: item.title.lower()):
        page_stem = f"wiki/{page_name_for_group(group)}"
        notes = group_notes.get(group.stem, [])
        parts.append(
            f"- {wikilink(page_stem)} from {wikilink(group.stem)} "
            f"({len(group.topics)} topics, {len(notes)} structured memories)"
        )
    if not groups:
        parts.append("- No topic groups found. Run topic regroup first.")
    parts.append("")
    return "\n".join(parts)


def write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def is_generated_wiki(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return False
    return 'generated: "ragmemory-wiki"' in text or "generated: ragmemory-wiki" in text


def clean_stale_wiki(wiki_dir: Path, expected: set[Path]) -> int:
    if not wiki_dir.exists():
        return 0
    removed = 0
    for path in wiki_dir.glob("*.md"):
        if path not in expected and is_generated_wiki(path):
            path.unlink()
            removed += 1
    return removed


def generate_wiki(
    obsidian_path: Path,
    max_topics: int,
    max_items_per_section: int,
    llm_options: WikiLLMOptions | None = None,
    llm_max_notes: int = 30,
    include_forgotten: bool = False,
) -> dict[str, int]:
    groups = load_topic_groups(obsidian_path)
    structured = load_structured_notes(obsidian_path, include_forgotten=include_forgotten)
    group_notes = {
        group.stem: [note for note in structured if group.stem in note.topic_groups]
        for group in groups
    }
    wiki_dir = obsidian_path / "wiki"
    expected: set[Path] = set()
    written = 0
    llm_written = 0
    llm_skipped = 0
    llm_cache_path = wiki_dir / ".llm_cache.json"
    llm_cache = load_llm_cache(llm_cache_path)
    llm_client = build_llm_client(llm_options) if llm_options and llm_options.enabled else None
    llm_remaining = llm_options.limit if llm_options and llm_options.limit is not None else None
    index_path = wiki_dir / "index.md"
    expected.add(index_path)
    if write_if_changed(index_path, render_index(groups, group_notes)):
        written += 1
    for group in groups:
        path = wiki_dir / f"{page_name_for_group(group)}.md"
        expected.add(path)
        notes = group_notes.get(group.stem, [])
        llm_summary = None
        if llm_options and llm_options.enabled and llm_client is not None:
            cache_key = llm_cache_key(group, notes, llm_max_notes, llm_client.options.model)
            if not llm_options.force and cache_key in llm_cache:
                llm_summary = llm_cache[cache_key]
                llm_skipped += 1
            elif llm_remaining is None or llm_remaining > 0:
                llm_summary = generate_llm_summary(group, notes, llm_client, llm_options, llm_max_notes)
                if llm_summary:
                    llm_cache[cache_key] = llm_summary
                    llm_written += 1
                if llm_remaining is not None:
                    llm_remaining -= 1
        if write_if_changed(
            path,
            render_group_page(
                group,
                notes,
                max_topics=max_topics,
                max_items_per_section=max_items_per_section,
                llm_summary=llm_summary,
            ),
        ):
            written += 1
    if llm_options and llm_options.enabled:
        save_llm_cache(llm_cache_path, llm_cache)
    removed = clean_stale_wiki(wiki_dir, expected)
    return {
        "groups": len(groups),
        "structured": len(structured),
        "written": written,
        "removed": removed,
        "llm_written": llm_written,
        "llm_cached": llm_skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a wiki from the Obsidian memory graph.")
    parser.add_argument("--obsidian", type=Path, default=DEFAULT_OBSIDIAN_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-topics", type=int, default=80)
    parser.add_argument("--max-items-per-section", type=int, default=40)
    parser.add_argument("--include-forgotten", action="store_true")
    parser.add_argument("--llm", action="store_true", help="Add cached LLM summaries to wiki pages.")
    parser.add_argument("--llm-force", action="store_true", help="Regenerate LLM summaries even when cached.")
    parser.add_argument("--llm-limit", type=int, help="Maximum number of uncached LLM pages to generate this run.")
    parser.add_argument("--llm-max-notes", type=int, default=30, help="Structured memory summaries sent per group.")
    args = parser.parse_args()
    load_config(args.config)
    llm_options = wiki_llm_options_from_env(
        enabled=args.llm,
        force=args.llm_force,
        limit=args.llm_limit,
    )
    stats = generate_wiki(
        args.obsidian,
        max_topics=max(1, args.max_topics),
        max_items_per_section=max(1, args.max_items_per_section),
        llm_options=llm_options,
        llm_max_notes=max(1, args.llm_max_notes),
        include_forgotten=args.include_forgotten,
    )
    print(
        f"Generated wiki for {stats['groups']} topic group(s), "
        f"{stats['structured']} structured note(s). "
        f"Written: {stats['written']} | Removed stale: {stats['removed']} | "
        f"LLM generated: {stats['llm_written']} | LLM cached: {stats['llm_cached']} | "
        f"Output: {args.obsidian / 'wiki'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
