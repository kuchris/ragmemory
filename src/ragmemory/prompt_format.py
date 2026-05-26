from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ContextBundle, StructuredMemoryObject


def format_for_prompt(bundle: ContextBundle) -> str:
    parts = []
    if bundle.structured:
        parts.append("=== Structured Memory ===\n" + "\n---\n".join(
            _format_structured_object(obj) for obj in bundle.structured
        ))
    seen_texts: set[str] = set()
    if bundle.kept:
        relevant_texts = []
        for chunk in bundle.kept:
            text = _clean_prompt_memory_text(chunk.text)
            if not _should_include_prompt_memory_text(text):
                continue
            key = _dedupe_prompt_memory_key(text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            relevant_texts.append(text)
    else:
        relevant_texts = []
    if relevant_texts:
        parts.append("=== Relevant Memory ===\n" + "\n---\n".join(
            relevant_texts
        ))
    if bundle.recent:
        recent_lines = []
        for message in bundle.recent:
            text = _clean_prompt_memory_text(message.text)
            if not _should_include_prompt_memory_text(text):
                continue
            key = _dedupe_prompt_memory_key(text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            recent_lines.append(f"{message.role.upper()}: {text}")
        if recent_lines:
            parts.append("=== Recent Conversation ===\n" + "\n".join(recent_lines))
    return "\n\n".join(parts)


def _clean_prompt_memory_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(
        r"(?is)# Context from my IDE setup:\s*.*?## My request for Codex:\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?im)^## Active file:.*(?:\n|$)", "", cleaned)
    cleaned = re.sub(r"(?ims)^## Open tabs:\s*(?:\n- .*)+", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _should_include_prompt_memory_text(text: str) -> bool:
    normalized = _dedupe_prompt_memory_key(text)
    if not normalized:
        return False
    low_value = {
        "btw",
        "ok",
        "wait let me ask claude",
        "let me ask claude",
    }
    return normalized not in low_value


def _dedupe_prompt_memory_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _format_structured_object(obj: StructuredMemoryObject) -> str:
    tags = ", ".join(obj.tags)
    return (
        f"[{obj.type} | importance={obj.importance} | message_id={obj.message_id}]\n"
        f"Summary: {obj.summary}\n"
        f"Tags: {tags}\n"
        f"Source: {obj.source_text}"
    )
