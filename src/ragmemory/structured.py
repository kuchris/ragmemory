from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .llm import DEFAULT_LLM_PROVIDER, LLMProviderClient, LLMProviderOptions
from .models import StructuredMemoryObject

STRUCTURED_TOP_K = 3
STRUCTURED_MEMORY_MODEL = os.environ.get(
    "STRUCTURED_MEMORY_MODEL", "minimaxai/minimax-m2.7"
)
DEFAULT_STRUCTURED_MAX_CHARS = 6000
DEFAULT_STRUCTURED_MAX_TOKENS = 900
STRUCTURED_TYPES = {
    "decision", "preference", "constraint", "config", "table",
    "code_reference", "chart", "open_question",
}


def _strip_inline_comment(value: str) -> str:
    text = value.strip()
    for marker in (" #", "\t#", " ;", "\t;"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(_strip_inline_comment(value))
    except ValueError:
        return default


@dataclass
class StructuredExtractionOptions:
    max_chars: int = DEFAULT_STRUCTURED_MAX_CHARS
    max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS

    @classmethod
    def from_env(cls) -> "StructuredExtractionOptions":
        return cls(
            max_chars=max(0, _env_int("RAGMEMORY_STRUCTURED_MAX_CHARS", DEFAULT_STRUCTURED_MAX_CHARS)),
            max_tokens=max(1, _env_int("RAGMEMORY_STRUCTURED_MAX_TOKENS", DEFAULT_STRUCTURED_MAX_TOKENS)),
        )


class StructuredMemoryExtractor:
    def __init__(
        self,
        options: LLMProviderOptions | None = None,
        extraction_options: StructuredExtractionOptions | None = None,
    ):
        provider = os.environ.get("RAGMEMORY_STRUCTURED_PROVIDER", DEFAULT_LLM_PROVIDER)
        self.llm = LLMProviderClient(
            options or LLMProviderOptions.from_env(provider, STRUCTURED_MEMORY_MODEL)
        )
        self.options = extraction_options or StructuredExtractionOptions.from_env()

    def extract(self, role: str, text: str, message_id: int) -> list[StructuredMemoryObject]:
        if not text.strip():
            return []
        if not self.llm.options.api_key:
            return []

        prompt = self._build_prompt(role, text)
        content = self.llm.complete_chat(
            messages=[
                {"role": "system", "content": "Extract only durable memory objects. Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=self.options.max_tokens,
        )
        if content is None:
            print(
                f"  [structured skipped] {self.llm.options.provider} extraction failed: "
                f"{self.llm.last_error}"
            )
            return []

        data = self._parse_json(content)
        objects = data.get("objects", []) if isinstance(data, dict) else []
        return [obj for item in objects if (obj := self._coerce_object(item, role, message_id))]

    def _build_prompt(self, role: str, text: str) -> str:
        message_text = text if self.options.max_chars <= 0 else text[:self.options.max_chars]
        return f"""Role: {role}

Allowed types:
- decision
- preference
- constraint
- config
- table
- code_reference
- chart
- open_question

Extract only information that should be remembered across sessions.
Exact fenced code/config/chart blocks and Markdown tables are extracted by code before this step.
Only return config/table/code_reference/chart if there is a durable artifact that was not already obvious as a fenced block or Markdown table.
If nothing durable exists, return {{"objects": []}}.

Tag rules:
- Tags must be concrete subject labels: project names, feature names, product/tool names, repo/module names, or durable workflow concepts.
- Prefer 1-4 specific tags per object.
- Do not copy the object type into tags.
- Do not use generic artifact/meta/language tags such as code_reference, config, decision, preference, table, chart, text, profile, python, powershell, json, yaml, markdown, important, context, memory, note.
- Good examples: ragmemory, obsidian-export, codex-hooks, memory-decay, topic-filtering.

Return JSON in this exact shape:
{{
  "objects": [
    {{
      "type": "decision",
      "summary": "short durable memory",
      "source_text": "exact supporting text",
      "tags": ["short", "lowercase", "tags"],
      "importance": 0.8
    }}
  ]
}}

Message:
{message_text}
"""

    def _parse_json(self, content: str) -> dict:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    def _coerce_object(
        self, item: object, role: str, message_id: int
    ) -> StructuredMemoryObject | None:
        if not isinstance(item, dict):
            return None
        obj_type = str(item.get("type", "")).strip().lower()
        if obj_type not in STRUCTURED_TYPES:
            return None
        summary = str(item.get("summary", "")).strip()
        source_text = str(item.get("source_text", "")).strip()
        if not summary or not source_text:
            return None
        raw_tags = item.get("tags", [])
        tags = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]
        try:
            importance = float(item.get("importance", 0.7))
        except (TypeError, ValueError):
            importance = 0.7
        importance = round(max(0.0, min(importance, 1.0)), 3)
        content_hash = str(item.get("content_hash", "")).strip() or None
        return StructuredMemoryObject(
            id=f"sm_{uuid.uuid4()}",
            type=obj_type,
            summary=summary,
            source_text=source_text,
            tags=tags[:8],
            importance=importance,
            message_id=message_id,
            role=role,
            content_hash=content_hash,
        )


class StructuredMemoryStore:
    def __init__(self, path: Path, collection):
        self._path = path
        self.collection = collection
        self.objects: dict[str, StructuredMemoryObject] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            obj = StructuredMemoryObject(**data)
            self.objects[obj.id] = obj

    def add_many(self, objects: list[StructuredMemoryObject]):
        if not objects:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for obj in objects:
                self.objects[obj.id] = obj
                f.write(json.dumps(obj.__dict__, ensure_ascii=False) + "\n")
        self._add_to_collection(objects)

    def rebuild_index(self) -> int:
        if self.collection.count() > 0:
            data = self.collection.get()
            if data["ids"]:
                self.collection.delete(ids=data["ids"])
        objects = list(self.objects.values())
        if not objects:
            return 0
        self._add_to_collection(objects)
        return len(objects)

    def _add_to_collection(self, objects: list[StructuredMemoryObject]) -> None:
        self.collection.add(
            ids=[obj.id for obj in objects],
            documents=[self._document_text(obj) for obj in objects],
            metadatas=[
                {
                    "type": obj.type,
                    "message_id": obj.message_id,
                    "role": obj.role,
                    "importance": obj.importance,
                    "tags": ",".join(obj.tags),
                }
                for obj in objects
            ],
        )

    def search(self, query: str, top_k: int = STRUCTURED_TOP_K) -> list[StructuredMemoryObject]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas"],
        )
        found = []
        for obj_id in results["ids"][0]:
            obj = self.objects.get(obj_id)
            if obj:
                found.append(obj)
        return found

    def _document_text(self, obj: StructuredMemoryObject) -> str:
        return (
            f"Type: {obj.type}\n"
            f"Summary: {obj.summary}\n"
            f"Tags: {', '.join(obj.tags)}\n"
            f"Source: {obj.source_text}"
        )

    def __len__(self):
        return len(self.objects)
