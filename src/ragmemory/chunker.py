from __future__ import annotations

import re
import uuid

from .models import Chunk, score_importance

CHUNK_MAX_TOKENS = 300
CHUNK_MIN_TOKENS = 80
HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


class Chunker:
    def split(self, text: str, message_id: int, role: str) -> list[Chunk]:
        paragraphs = self._split_with_headers(text)
        raw = []
        for para, header in paragraphs:
            injected = f"[{header}] {para}" if header else para
            if len(injected) / 4 > CHUNK_MAX_TOKENS:
                raw.extend(self._split_sentences(injected, message_id, role))
            else:
                raw.append(self._chunk(injected, message_id, role))

        merged = []
        for c in raw:
            if merged and len(merged[-1].text) / 4 < CHUNK_MIN_TOKENS:
                merged[-1].text += " " + c.text
                merged[-1].importance = score_importance(merged[-1].text)
            else:
                c.importance = score_importance(c.text)
                merged.append(c)
        return merged

    def _split_with_headers(self, text: str) -> list[tuple[str, str | None]]:
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        result, current_header = [], None
        for para in paragraphs:
            m = HEADER_RE.match(para)
            if m:
                current_header = m.group(1).strip()
                result.append((para, None))
            else:
                result.append((para, current_header))
        return result

    def _split_sentences(self, text: str, message_id: int, role: str) -> list[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            candidate = (current + " " + sent).strip()
            if len(candidate) / 4 > CHUNK_MAX_TOKENS and current:
                chunks.append(self._chunk(current, message_id, role))
                current = sent
            else:
                current = candidate
        if current:
            chunks.append(self._chunk(current, message_id, role))
        return chunks

    def _chunk(self, text: str, message_id: int, role: str) -> Chunk:
        return Chunk(id=str(uuid.uuid4()), text=text, message_id=message_id, role=role)
