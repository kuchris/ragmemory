from __future__ import annotations

import json
from pathlib import Path

from .models import LedgerEntry, RetrievedChunk

MISSING_CTX_PHRASES = [
    "as we said", "earlier", "we decided", "you mentioned", "what did we",
    "continue", "the thing we", "remind me", "what was",
]


class RemovalLedger:
    def __init__(self, path: Path):
        self._path = path
        self.entries: list[LedgerEntry] = []
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.entries = [
                LedgerEntry(
                    chunk_id=e["chunk_id"],
                    text=e["text"],
                    importance=e["importance"],
                    message_id=e.get("message_id", e.get("turn_id", 0)),
                    reason=e.get("reason", "budget"),
                )
                for e in data
            ]

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([e.__dict__ for e in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log(self, chunk: RetrievedChunk, reason: str = "budget"):
        self.entries.append(
            LedgerEntry(chunk.id, chunk.text, chunk.importance, chunk.message_id, reason)
        )
        self._save()
        print(f"  [ledger +1] importance={chunk.importance} | {chunk.text[:60]}...")

    def search(self, query: str, top_k: int = 3) -> list[LedgerEntry]:
        query_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            overlap = len(query_words & set(entry.text.lower().split()))
            if overlap > 0:
                scored.append((overlap + entry.importance, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def looks_like_missing_context(self, query: str) -> bool:
        return any(phrase in query.lower() for phrase in MISSING_CTX_PHRASES)

    def __len__(self):
        return len(self.entries)
