from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = False

    def build(self, ids: list[str], texts: list[str]):
        self._ids = list(ids)
        self._tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._dirty = False

    def add(self, doc_id: str, text: str):
        self._ids.append(doc_id)
        self._tokenized.append(text.lower().split())
        self._dirty = True

    def clear(self):
        self._ids = []
        self._tokenized = []
        self._bm25 = None
        self._dirty = False

    def search(self, query: str, top_k: int) -> list[str]:
        if not self._ids or top_k <= 0:
            return []
        if self._dirty or self._bm25 is None:
            self._bm25 = BM25Okapi(self._tokenized)
            self._dirty = False
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._ids[i] for i in ranked[:top_k] if scores[i] > 0]

    def __len__(self):
        return len(self._ids)
