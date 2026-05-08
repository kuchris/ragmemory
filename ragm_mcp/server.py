import re
import uuid
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi
from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "chroma_db"
CHUNK_MAX_TOKENS = 300
CHUNK_MIN_TOKENS = 80
RECENT_MESSAGES = 12
RETRIEVE_TOP_K = 5
CONTEXT_TOKEN_BUDGET = 2000
RRF_K = 60

HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
HIGH_SIGNAL_WORDS = {
    "decided", "decision", "must", "never", "always", "constraint",
    "requirement", "critical", "important", "warning", "error", "fix",
    "bug", "todo", "note", "remember",
}
MISSING_CTX_PHRASES = [
    "as we said", "earlier", "we decided", "you mentioned", "what did we",
    "continue", "the thing we", "remind me", "what was",
]

# ── Importance scoring ────────────────────────────────────────────────────────

def score_importance(text: str) -> float:
    score = 0.5
    score += min(len(text.split()) / 80, 0.15)
    score += 0.15 * any(w in text.lower() for w in HIGH_SIGNAL_WORDS)
    score += 0.10 * bool(re.search(r"```|def |class |import ", text))
    score += 0.05 * bool(re.search(r"\b\d+\.?\d*\b", text))
    return round(min(score, 1.0), 3)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    text: str
    message_id: int
    role: str
    importance: float = 0.5

@dataclass
class RetrievedChunk:
    id: str
    text: str
    importance: float
    message_id: int

@dataclass
class LedgerEntry:
    chunk_id: str
    text: str
    importance: float
    message_id: int
    reason: str = "budget"

# ── BM25 index ────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, doc_id: str, text: str):
        self._ids.append(doc_id)
        self._tokenized.append(text.lower().split())
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int) -> list[str]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._ids[i] for i in ranked[:top_k] if scores[i] > 0]

    def __len__(self):
        return len(self._ids)

# ── Removal ledger ────────────────────────────────────────────────────────────

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
        self._path.write_text(
            json.dumps([e.__dict__ for e in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log(self, chunk: RetrievedChunk, reason: str = "budget"):
        self.entries.append(
            LedgerEntry(chunk.id, chunk.text, chunk.importance, chunk.message_id, reason)
        )
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[LedgerEntry]:
        query_words = set(query.lower().split())
        scored = [
            (len(query_words & set(e.text.lower().split())) + e.importance, e)
            for e in self.entries
            if query_words & set(e.text.lower().split())
        ]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def looks_like_missing_context(self, query: str) -> bool:
        return any(phrase in query.lower() for phrase in MISSING_CTX_PHRASES)

    def __len__(self):
        return len(self.entries)

# ── Chunker ───────────────────────────────────────────────────────────────────

class Chunker:
    def split(self, text: str, message_id: int, role: str) -> list[Chunk]:
        raw = []
        for para, header in self._split_with_headers(text):
            injected = f"[{header}] {para}" if header else para
            if len(injected) / 4 > CHUNK_MAX_TOKENS:
                raw.extend(self._split_sentences(injected, message_id, role))
            else:
                raw.append(self._make(injected, message_id, role))

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
        result, current_header = [], None
        for para in [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]:
            m = HEADER_RE.match(para)
            if m:
                current_header = m.group(1).strip()
                result.append((para, None))
            else:
                result.append((para, current_header))
        return result

    def _split_sentences(self, text: str, message_id: int, role: str) -> list[Chunk]:
        chunks, current = [], ""
        for sent in re.split(r"(?<=[.!?])\s+", text):
            candidate = (current + " " + sent).strip()
            if len(candidate) / 4 > CHUNK_MAX_TOKENS and current:
                chunks.append(self._make(current, message_id, role))
                current = sent
            else:
                current = candidate
        if current:
            chunks.append(self._make(current, message_id, role))
        return chunks

    def _make(self, text: str, message_id: int, role: str) -> Chunk:
        return Chunk(id=str(uuid.uuid4()), text=text, message_id=message_id, role=role)

# ── Memory store ──────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self.state_file = DB_PATH / "state.json"
        self.embed_fn = DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_or_create_collection(
            "chat_memory", embedding_function=self.embed_fn
        )
        self.chunker = Chunker()
        self.ledger = RemovalLedger(DB_PATH / "ledger.json")
        self.bm25 = BM25Index()
        self.raw_log: list[dict] = []
        self.message_id = 0
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.raw_log = data.get("raw_log", [])
            self.message_id = data.get("message_id", data.get("turn_id", 0))
        if self.collection.count() > 0:
            all_docs = self.collection.get(include=["documents"])
            for doc_id, doc in zip(all_docs["ids"], all_docs["documents"]):
                self.bm25.add(doc_id, doc)

    def _save_state(self):
        self.state_file.write_text(
            json.dumps({"raw_log": self.raw_log, "message_id": self.message_id},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_message(self, role: str, text: str) -> int:
        self.raw_log.append({"role": role, "text": text, "message_id": self.message_id})
        chunks = self.chunker.split(text, self.message_id, role)
        if chunks:
            self.collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{"message_id": c.message_id, "role": c.role, "importance": c.importance}
                           for c in chunks],
            )
            for c in chunks:
                self.bm25.add(c.id, c.text)
        self.message_id += 1
        self._save_state()
        return len(chunks)

    def retrieve(self, query: str, top_k: int = RETRIEVE_TOP_K) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0:
            return []
        fetch_k = min(top_k * 2, count)

        emb_ids = self.collection.query(
            query_texts=[query], n_results=fetch_k, include=["metadatas"]
        )["ids"][0]
        emb_rank = {id: i for i, id in enumerate(emb_ids)}

        bm25_ids = self.bm25.search(query, top_k=fetch_k)
        bm25_rank = {id: i for i, id in enumerate(bm25_ids)}

        all_ids = list(set(emb_ids) | set(bm25_ids))
        rrf = {
            id: (1 / (RRF_K + emb_rank.get(id, 9999) + 1))
               + (1 / (RRF_K + bm25_rank.get(id, 9999) + 1))
            for id in all_ids
        }
        merged = sorted(all_ids, key=lambda x: -rrf[x])[:fetch_k]

        fetched = self.collection.get(ids=merged, include=["documents", "metadatas"])
        id_to_chunk = {
            id: RetrievedChunk(id=id, text=doc,
                               importance=meta.get("importance", 0.5),
                               message_id=meta.get("message_id", meta.get("turn_id", 0)))
            for id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
        }

        seen, unique = set(), []
        for id in merged:
            c = id_to_chunk.get(id)
            if c and c.text not in seen:
                seen.add(c.text)
                unique.append(c)
            if len(unique) == top_k:
                break
        return unique

    def build_context(self, query: str) -> str:
        recent = self.raw_log[-RECENT_MESSAGES:]
        recent_texts = {m["text"] for m in recent}

        retrieved = [r for r in self.retrieve(query) if r.text not in recent_texts]

        if self.ledger.looks_like_missing_context(query) and len(self.ledger) > 0:
            for e in self.ledger.search(query, top_k=2):
                if e.text not in recent_texts:
                    retrieved.append(RetrievedChunk(
                        id=e.chunk_id, text=e.text,
                        importance=e.importance, message_id=e.message_id,
                    ))

        budget = CONTEXT_TOKEN_BUDGET
        kept = []
        for chunk in sorted(retrieved, key=lambda c: -c.importance):
            tokens = len(chunk.text) // 4
            if budget - tokens >= 0:
                kept.append(chunk)
                budget -= tokens
            else:
                self.ledger.log(chunk, reason="budget")

        parts = []
        if kept:
            parts.append("=== Relevant Memory ===\n" + "\n---\n".join(c.text for c in kept))
        if recent:
            parts.append("=== Recent Conversation ===\n" +
                         "\n".join(f"{m['role'].upper()}: {m['text']}" for m in recent))
        return "\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "chunks": self.collection.count(),
            "next_message_id": self.message_id,
            "messages": len(self.raw_log),
            "bm25_indexed": len(self.bm25),
            "ledger_entries": len(self.ledger),
        }


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("RAG Memory")
store = MemoryStore()


@mcp.tool()
def recall(user_message: str) -> str:
    """Call at the START of every turn. Stores the user message and returns relevant memory context."""
    store.add_message("user", user_message)
    ctx = store.build_context(user_message)
    return ctx if ctx else "No relevant memory found."


@mcp.tool()
def save(summary: str) -> str:
    """Call at the END of every turn. Store a short summary of your response (not the full text)."""
    store.add_message("assistant", summary)
    return "Saved."


@mcp.tool()
def remember_document(text: str, role: str = "user") -> str:
    """Store a large document or long text by splitting it into chunks first."""
    normalized = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)
    n = store.add_message(role, normalized)
    return f"Stored {n} chunk(s). Total chunks: {store.collection.count()}"


@mcp.tool()
def memory_stats() -> str:
    """Get current memory statistics."""
    s = store.stats()
    return (f"Chunks: {s['chunks']} | Next message ID: {s['next_message_id']} | "
            f"Messages: {s['messages']} | Ledger: {s['ledger_entries']}")


if __name__ == "__main__":
    mcp.run()
