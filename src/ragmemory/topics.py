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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


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


@dataclass
class TopicRegroupOptions:
    provider: str
    model: str
    max_tokens: int = DEFAULT_TOPIC_REGROUP_MAX_TOKENS
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
            max_tokens=max(1, _env_int("RAGMEMORY_TOPIC_MAX_TOKENS", DEFAULT_TOPIC_REGROUP_MAX_TOKENS)),
            thinking=os.environ.get("RAGMEMORY_TOPIC_THINKING", "").strip().lower(),
        )


def topic_taxonomy_path(db_path: Path) -> Path:
    return db_path / TOPIC_TAXONOMY_FILE


def _topic_input_objects(objects: list[StructuredMemoryObject]) -> list[dict]:
    rows = []
    for obj in sorted(objects, key=lambda item: (item.message_id, item.id)):
        rows.append(
            {
                "id": obj.id,
                "type": obj.type,
                "summary": obj.summary,
                "tags": obj.tags,
                "importance": obj.importance,
                "message_id": obj.message_id,
            }
        )
    return rows


def validate_topic_taxonomy(data: dict, valid_structured_ids: set[str]) -> dict:
    if not isinstance(data, dict):
        raise ValueError("taxonomy must be a JSON object")
    topics = data.get("topics")
    if not isinstance(topics, list):
        raise ValueError("taxonomy.topics must be a list")

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

    return {
        "generated_at": str(data.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        "model": str(data.get("model") or ""),
        "source_object_count": int(data.get("source_object_count") or len(valid_structured_ids)),
        "topics": normalized_topics,
    }


def save_validated_topic_taxonomy(path: Path, data: dict, valid_structured_ids: set[str]) -> dict:
    validated = validate_topic_taxonomy(data, valid_structured_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return validated


def build_topic_regroup_prompt(objects: list[StructuredMemoryObject]) -> str:
    payload = _topic_input_objects(objects)
    return f"""Regroup RagMemory structured memories into semantic Obsidian topic hubs.

Input objects include only id, type, summary, tags, importance, and message_id.
Do not require a fixed number of topics. Create as many topics as are useful.
Merge near-duplicates and related tags into broader topics.
Avoid generic language/type topics such as python, powershell, text, config, decision, preference.

Return strict JSON only, with this shape:
{{
  "topics": [
    {{
      "id": "message-compaction",
      "title": "Message compaction",
      "description": "Compaction, compact_text, compact backfill, evidence repair.",
      "aliases": ["compact-backfill", "compact-text", "compact-status"],
      "structured_ids": ["sm_..."]
    }}
  ]
}}

Structured memories:
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
            {"role": "user", "content": build_topic_regroup_prompt(objects)},
        ],
        temperature=0,
        max_tokens=options.max_tokens,
    )
    if content is None:
        raise RuntimeError(client.last_error or "topic regroup LLM call failed")
    try:
        data = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"topic regroup returned invalid JSON: {exc}") from exc
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["model"] = llm_options.model
    data["source_object_count"] = len(objects)
    valid_ids = {obj.id for obj in objects}
    return save_validated_topic_taxonomy(topic_taxonomy_path(db_path), data, valid_ids)
