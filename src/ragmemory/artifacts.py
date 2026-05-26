from __future__ import annotations

import re
import uuid

from .models import StructuredMemoryObject, evidence_content_hash

FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<body>.*?)(?:\n)?```",
    re.DOTALL,
)
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


class ExactArtifactExtractor:
    CONFIG_LANGS = {"json", "toml", "yaml", "yml", "ini", "env", "xml"}

    def extract(self, role: str, text: str, message_id: int) -> list[StructuredMemoryObject]:
        objects = self._extract_fenced_blocks(role, text, message_id)
        objects.extend(self._extract_markdown_tables(role, text, message_id))
        return objects

    def _extract_fenced_blocks(
        self, role: str, text: str, message_id: int
    ) -> list[StructuredMemoryObject]:
        objects = []
        for match in FENCED_BLOCK_RE.finditer(text):
            source_text = match.group(0)
            lang = match.group("lang").lower()
            obj_type = self._type_for_fenced_lang(lang)
            objects.append(StructuredMemoryObject(
                id=f"sm_{uuid.uuid4()}",
                type=obj_type,
                summary=self._summary_for_artifact(obj_type, lang),
                source_text=source_text,
                tags=[tag for tag in [lang, obj_type] if tag],
                importance=0.85,
                message_id=message_id,
                role=role,
                content_hash=evidence_content_hash(source_text, lang or obj_type),
            ))
        return objects

    def _extract_markdown_tables(
        self, role: str, text: str, message_id: int
    ) -> list[StructuredMemoryObject]:
        lines = text.splitlines()
        objects = []
        i = 0
        while i < len(lines) - 1:
            if "|" not in lines[i] or not TABLE_SEPARATOR_RE.match(lines[i + 1]):
                i += 1
                continue
            start = i
            i += 2
            while i < len(lines) and "|" in lines[i].strip():
                i += 1
            source_text = "\n".join(lines[start:i])
            objects.append(StructuredMemoryObject(
                id=f"sm_{uuid.uuid4()}",
                type="table",
                summary="Exact Markdown table",
                source_text=source_text,
                tags=["markdown", "table"],
                importance=0.85,
                message_id=message_id,
                role=role,
                content_hash=evidence_content_hash(source_text, "table"),
            ))
        return objects

    def _type_for_fenced_lang(self, lang: str) -> str:
        if lang in self.CONFIG_LANGS:
            return "config"
        if lang == "mermaid":
            return "chart"
        return "code_reference"

    def _summary_for_artifact(self, obj_type: str, lang: str) -> str:
        if obj_type == "config":
            return f"Exact {lang or 'config'} config block"
        if obj_type == "chart":
            return "Exact Mermaid chart block"
        return f"Exact {lang or 'code'} block"
