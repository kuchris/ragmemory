from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .llm import DEFAULT_LLM_PROVIDER, LLMProviderClient, LLMProviderOptions
from .models import StructuredMemoryObject
from .structured import STRUCTURED_MEMORY_MODEL


TOPIC_TAXONOMY_FILE = "topic_taxonomy.json"
JOB_TYPE_TOPIC_REGROUP = "topic_regroup"
DEFAULT_TOPIC_REGROUP_MAX_TOKENS = 6000
DEFAULT_TOPIC_REGROUP_MAX_INPUT_TOPICS = 150
DEFAULT_TOPIC_REGROUP_MIN_GROUPS = 10


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "topic"


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    stripped = _strip_json_fence(text)
    if not stripped:
        raise ValueError("topic regroup returned empty response")
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        preview = stripped[:200].replace("\n", " ")
        raise ValueError(f"topic regroup response did not contain a JSON object: {preview}")
    return stripped[start:end + 1]


def _topic_json_repair_prompt(content: str, error: str) -> str:
    return f"""Repair this malformed topic taxonomy JSON.

The parser error was: {error}

Rules:
- Return strict JSON only.
- Keep the same schema with a top-level "groups" array.
- Preserve valid group ids, titles, descriptions, aliases, and topic_ids when possible.
- Do not add commentary, markdown fences, trailing commas, or comments.

Malformed JSON:
{content}
"""


def _load_topic_response_json(
    content: str,
    client: LLMProviderClient | None = None,
    repair_max_tokens: int = DEFAULT_TOPIC_REGROUP_MAX_TOKENS,
) -> dict:
    try:
        return json.loads(_extract_json_object(content))
    except json.JSONDecodeError as first_exc:
        if client is None:
            raise ValueError(f"topic regroup returned invalid JSON: {first_exc}") from first_exc
        repaired = client.complete_chat(
            messages=[
                {
                    "role": "system",
                    "content": "You repair malformed JSON. Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": _topic_json_repair_prompt(content, str(first_exc)),
                },
            ],
            temperature=0,
            max_tokens=repair_max_tokens,
        )
        if repaired is None:
            raise ValueError(
                "topic regroup returned invalid JSON and repair failed: "
                f"{client.last_error or first_exc}"
            ) from first_exc
        try:
            return json.loads(_extract_json_object(repaired))
        except json.JSONDecodeError as second_exc:
            raise ValueError(
                "topic regroup returned invalid JSON; repair response was also invalid: "
                f"{second_exc}"
            ) from second_exc


@dataclass
class TopicRegroupOptions:
    provider: str
    model: str
    enabled: bool = True
    max_tokens: int = DEFAULT_TOPIC_REGROUP_MAX_TOKENS
    max_input_topics: int = DEFAULT_TOPIC_REGROUP_MAX_INPUT_TOPICS
    min_groups: int = DEFAULT_TOPIC_REGROUP_MIN_GROUPS
    thinking: str = ""

    @classmethod
    def from_env(cls) -> "TopicRegroupOptions":
        provider = os.environ.get(
            "RAGMEMORY_TOPIC_PROVIDER",
            os.environ.get("RAGMEMORY_STRUCTURED_PROVIDER", DEFAULT_LLM_PROVIDER),
        ).strip().lower()
        model = os.environ.get("RAGMEMORY_TOPIC_MODEL", STRUCTURED_MEMORY_MODEL).strip()
        return cls(
            provider=provider,
            model=model,
            enabled=_env_bool("RAGMEMORY_TOPIC_ENABLE", True),
            max_tokens=max(1, _env_int("RAGMEMORY_TOPIC_MAX_TOKENS", DEFAULT_TOPIC_REGROUP_MAX_TOKENS)),
            max_input_topics=max(
                1,
                _env_int("RAGMEMORY_TOPIC_MAX_INPUT_TOPICS", DEFAULT_TOPIC_REGROUP_MAX_INPUT_TOPICS),
            ),
            min_groups=max(1, _env_int("RAGMEMORY_TOPIC_MIN_GROUPS", DEFAULT_TOPIC_REGROUP_MIN_GROUPS)),
            thinking=os.environ.get("RAGMEMORY_TOPIC_THINKING", "").strip().lower(),
        )


def topic_taxonomy_path(db_path: Path) -> Path:
    return db_path / TOPIC_TAXONOMY_FILE


def _topic_title(topic_id: str) -> str:
    return topic_id.replace("-", " ").strip().capitalize() or "Topic"


def _topic_input_candidates(objects: list[StructuredMemoryObject], max_topics: int) -> list[dict]:
    candidates: dict[str, dict] = {}
    for obj in sorted(objects, key=lambda item: (item.message_id, item.id)):
        seen_tags = set()
        for tag in obj.tags:
            topic_id = _slug(tag)
            if not topic_id or topic_id in seen_tags:
                continue
            seen_tags.add(topic_id)
            item = candidates.setdefault(
                topic_id,
                {
                    "id": topic_id,
                    "title": str(tag).strip() or _topic_title(topic_id),
                    "count": 0,
                    "object_types": set(),
                    "sample_summaries": [],
                    "max_importance": 0.0,
                },
            )
            item["count"] += 1
            item["object_types"].add(obj.type)
            item["max_importance"] = max(item["max_importance"], float(obj.importance))
            if len(item["sample_summaries"]) < 3 and obj.summary.strip():
                item["sample_summaries"].append(obj.summary.strip())

    ranked = sorted(
        candidates.values(),
        key=lambda item: (-item["count"], -item["max_importance"], item["id"]),
    )
    rows = []
    for item in ranked[:max_topics]:
        rows.append(
            {
                "id": item["id"],
                "title": item["title"],
                "count": item["count"],
                "object_types": sorted(item["object_types"]),
                "sample_summaries": item["sample_summaries"],
            }
        )
    return rows


def validate_topic_taxonomy(
    data: dict,
    valid_structured_ids: set[str],
    valid_topic_ids: set[str] | None = None,
) -> dict:
    if not isinstance(data, dict):
        raise ValueError("taxonomy must be a JSON object")
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        raise ValueError("taxonomy.topics must be a list")
    groups = data.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("taxonomy.groups must be a list")

    normalized_topics = []
    seen_ids = set()
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise ValueError(f"topic {index} must be an object")
        topic_id = _slug(str(topic.get("id") or topic.get("title") or ""))
        title = str(topic.get("title") or "").strip()
        description = str(topic.get("description") or "").strip()
        aliases = topic.get("aliases", [])
        structured_ids = topic.get("structured_ids", [])
        if not topic_id:
            raise ValueError(f"topic {index} missing id")
        if topic_id in seen_ids:
            raise ValueError(f"duplicate topic id: {topic_id}")
        if not title:
            raise ValueError(f"topic {topic_id} missing title")
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise ValueError(f"topic {topic_id} aliases must be a string list")
        if not isinstance(structured_ids, list) or not all(isinstance(item, str) for item in structured_ids):
            raise ValueError(f"topic {topic_id} structured_ids must be a string list")
        unknown_ids = sorted(set(structured_ids) - valid_structured_ids)
        if unknown_ids:
            raise ValueError(f"topic {topic_id} references unknown structured ids: {unknown_ids[:5]}")
        seen_ids.add(topic_id)
        normalized_topics.append(
            {
                "id": topic_id,
                "title": title,
                "description": description,
                "aliases": sorted(set(item.strip() for item in aliases if item.strip())),
                "structured_ids": sorted(set(structured_ids)),
            }
        )

    normalized_groups = []
    seen_group_ids = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"group {index} must be an object")
        group_id = _slug(str(group.get("id") or group.get("title") or ""))
        title = str(group.get("title") or "").strip()
        description = str(group.get("description") or "").strip()
        aliases = group.get("aliases", [])
        topic_ids = group.get("topic_ids", group.get("topics", []))
        if not group_id:
            raise ValueError(f"group {index} missing id")
        if group_id in seen_group_ids:
            raise ValueError(f"duplicate group id: {group_id}")
        if not title:
            raise ValueError(f"group {group_id} missing title")
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise ValueError(f"group {group_id} aliases must be a string list")
        if not isinstance(topic_ids, list) or not all(isinstance(item, str) for item in topic_ids):
            raise ValueError(f"group {group_id} topic_ids must be a string list")
        normalized_topic_ids = sorted(set(_slug(item) for item in topic_ids if _slug(item)))
        if valid_topic_ids is not None:
            unknown_topics = sorted(set(normalized_topic_ids) - valid_topic_ids)
            if unknown_topics:
                raise ValueError(f"group {group_id} references unknown topic ids: {unknown_topics[:5]}")
        seen_group_ids.add(group_id)
        normalized_groups.append(
            {
                "id": group_id,
                "title": title,
                "description": description,
                "aliases": sorted(set(item.strip() for item in aliases if item.strip())),
                "topic_ids": normalized_topic_ids,
            }
        )

    return {
        "generated_at": str(data.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        "model": str(data.get("model") or ""),
        "source_object_count": int(data.get("source_object_count") or len(valid_structured_ids)),
        "source_topic_count": int(data.get("source_topic_count") or (len(valid_topic_ids) if valid_topic_ids else 0)),
        "topics": normalized_topics,
        "groups": normalized_groups,
    }


def save_validated_topic_taxonomy(
    path: Path,
    data: dict,
    valid_structured_ids: set[str],
    valid_topic_ids: set[str] | None = None,
) -> dict:
    validated = validate_topic_taxonomy(data, valid_structured_ids, valid_topic_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return validated


def build_topic_regroup_prompt(
    objects: list[StructuredMemoryObject],
    max_input_topics: int = DEFAULT_TOPIC_REGROUP_MAX_INPUT_TOPICS,
    min_groups: int = DEFAULT_TOPIC_REGROUP_MIN_GROUPS,
) -> str:
    payload = _topic_input_candidates(objects, max_input_topics)
    return f"""Group RagMemory leaf topics into a higher-level Obsidian topic map.

Input rows are existing leaf topics, not raw memories. Each row includes topic id, title, object count, object types, and a few sample summaries.
Do not remove, replace, or rename the input leaf topics.
Create upper-layer groups that contain related topic_ids.
Create at least {min_groups} groups when there are enough distinct leaf topics.
It is okay to leave narrow or unclear topic_ids ungrouped.
Avoid generic language/type groups such as python, powershell, text, config, decision, preference.
Every object in an array must be comma-separated.

Return strict JSON only, with this shape:
{{
  "groups": [
    {{
      "id": "obsidian-export",
      "title": "Obsidian export",
      "description": "Generated Obsidian mirror, graph colors, topic hubs, and export checks.",
      "aliases": ["obsidian-mirror"],
      "topic_ids": ["obsidian", "obsidian-export", "topic-hubs"]
    }}
  ]
}}

Leaf topics:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_topic_llm_options(options: TopicRegroupOptions) -> LLMProviderOptions:
    llm_options = LLMProviderOptions.from_env(options.provider, options.model)
    if options.thinking:
        llm_options.extra_body = LLMProviderOptions._extra_body_for(
            llm_options.provider,
            options.thinking,
        )
    return llm_options


def regroup_topics_with_llm(
    db_path: Path,
    objects: list[StructuredMemoryObject],
    options: TopicRegroupOptions | None = None,
) -> dict:
    options = options or TopicRegroupOptions.from_env()
    llm_options = build_topic_llm_options(options)
    client = LLMProviderClient(llm_options)
    content = client.complete_chat(
        messages=[
            {
                "role": "system",
                "content": "You curate concise topic taxonomies for a local coding memory graph. Return strict JSON only.",
            },
            {
                "role": "user",
                "content": build_topic_regroup_prompt(
                    objects,
                    options.max_input_topics,
                    options.min_groups,
                ),
            },
        ],
        temperature=0,
        max_tokens=options.max_tokens,
    )
    if content is None:
        raise RuntimeError(client.last_error or "topic regroup LLM call failed")
    data = _load_topic_response_json(content, client, options.max_tokens)
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["model"] = llm_options.model
    data["source_object_count"] = len(objects)
    topic_candidates = _topic_input_candidates(objects, options.max_input_topics)
    data["source_topic_count"] = len(topic_candidates)
    min_groups = min(options.min_groups, len(topic_candidates))
    groups = data.get("groups", [])
    if len(topic_candidates) >= options.min_groups and (
        not isinstance(groups, list) or len(groups) < min_groups
    ):
        raise ValueError(
            f"topic regroup returned {len(groups) if isinstance(groups, list) else 0} group(s); "
            f"expected at least {min_groups}"
        )
    valid_ids = {obj.id for obj in objects}
    valid_topic_ids = {item["id"] for item in topic_candidates}
    return save_validated_topic_taxonomy(topic_taxonomy_path(db_path), data, valid_ids, valid_topic_ids)
