import os
import re
import uuid
import json
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

CHUNK_MAX_TOKENS = 300
CHUNK_MIN_TOKENS = 80
RECENT_MESSAGES = 12
RETRIEVE_TOP_K = 5
STRUCTURED_TOP_K = 3
CONTEXT_TOKEN_BUDGET = 2000
RRF_K = 60  # reciprocal rank fusion constant
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
STRUCTURED_MEMORY_MODEL = os.environ.get(
    "STRUCTURED_MEMORY_MODEL", "meta/llama-3.1-8b-instruct"
)
STRUCTURED_TYPES = {
    "decision", "preference", "constraint", "config", "table",
    "code_reference", "chart", "open_question",
}
EXACT_ARTIFACT_TYPES = {"config", "table", "chart"}

HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<body>.*?)(?:\n)?```",
    re.DOTALL,
)
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
MISSING_CTX_PHRASES = [
    "as we said", "earlier", "we decided", "you mentioned", "what did we",
    "continue", "the thing we", "remind me", "what was",
]


# ── Importance scoring ────────────────────────────────────────────────────────

def score_importance(text: str) -> float:
    """Compatibility hook for old callers; raw chunk retention is score-based."""
    return 0.5


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
    score: float = 0.0


@dataclass
class LedgerEntry:
    chunk_id: str
    text: str
    importance: float
    message_id: int
    reason: str = "budget"


@dataclass
class StructuredMemoryObject:
    id: str
    type: str
    summary: str
    source_text: str
    tags: list[str]
    importance: float
    message_id: int
    role: str


@dataclass
class AddMessageResult:
    saved: bool
    deduped: bool
    message_id: int | None
    content_hash: str
    chunk_ids: list[str]
    structured_object_ids: list[str]
    queued_job_ids: list[str]


# ── BM25 index ────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, ids: list[str], texts: list[str]):
        self._ids = list(ids)
        self._tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def add(self, doc_id: str, text: str):
        self._ids.append(doc_id)
        self._tokenized.append(text.lower().split())
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int) -> list[str]:
        if not self._bm25 or not self._ids:
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


# ── Chunker ───────────────────────────────────────────────────────────────────

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


class StructuredMemoryExtractor:
    def __init__(self):
        api_key = os.environ.get(NVIDIA_API_KEY_ENV)
        self.client = None
        if api_key:
            from openai import OpenAI

            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )

    def extract(self, role: str, text: str, message_id: int) -> list[StructuredMemoryObject]:
        if not self.client or not text.strip():
            return []

        prompt = self._build_prompt(role, text)
        try:
            response = self.client.chat.completions.create(
                model=STRUCTURED_MEMORY_MODEL,
                messages=[
                    {"role": "system", "content": "Extract only durable memory objects. Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=900,
            )
        except Exception as exc:
            print(f"  [structured skipped] NVIDIA extraction failed: {exc}")
            return []

        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        objects = data.get("objects", []) if isinstance(data, dict) else []
        return [obj for item in objects if (obj := self._coerce_object(item, role, message_id))]

    def _build_prompt(self, role: str, text: str) -> str:
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
{text[:6000]}
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
        return StructuredMemoryObject(
            id=f"sm_{uuid.uuid4()}",
            type=obj_type,
            summary=summary,
            source_text=source_text,
            tags=tags[:8],
            importance=importance,
            message_id=message_id,
            role=role,
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


# ── Memory store ──────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, db_path: str = "./.data/chroma_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.state_db = self.db_path / "state.sqlite"
        self.state_file = self.db_path / "state.json"

        print("Loading embedding model...")
        self.embed_fn = DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            "chat_memory", embedding_function=self.embed_fn
        )
        self.structured_collection = self.client.get_or_create_collection(
            "structured_memory", embedding_function=self.embed_fn
        )
        self.chunker = Chunker()
        self.ledger = RemovalLedger(self.db_path / "ledger.json")
        self.structured = StructuredMemoryStore(
            self.db_path / "structured_memory.jsonl",
            self.structured_collection,
        )
        self.artifact_extractor = ExactArtifactExtractor()
        self.structured_extractor = StructuredMemoryExtractor()
        self.bm25 = BM25Index()
        self.raw_log: list[dict] = []
        self._message_hashes: dict[tuple[str, str], int] = {}
        self.message_id = 0

        self._load_state()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self):
        self._init_state_db()
        self._migrate_json_state()
        with sqlite3.connect(self.state_db) as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, text, content_hash
                FROM messages
                WHERE tombstoned = 0
                ORDER BY message_id
                """
            ).fetchall()
            self.raw_log = [
                {
                    "role": role,
                    "text": text,
                    "message_id": message_id,
                    "content_hash": content_hash,
                }
                for message_id, role, text, content_hash in rows
            ]
            next_id = conn.execute(
                "SELECT value FROM meta WHERE key = 'next_message_id'"
            ).fetchone()

        for message in self.raw_log:
            self._message_hashes[(message["role"], message["content_hash"])] = message["message_id"]

        if next_id:
            self.message_id = int(next_id[0])
        elif self.raw_log:
            self.message_id = max(message["message_id"] for message in self.raw_log) + 1

        if self.raw_log:
            print(f"  [restored] {len(self.raw_log)} messages, next message {self.message_id}")

        # rebuild BM25 from chromadb (always, since BM25 is in-memory)
        if self.collection.count() > 0:
            all_docs = self.collection.get(include=["documents"])
            self.bm25.build(all_docs["ids"], all_docs["documents"])
            print(f"  [bm25] rebuilt index with {len(self.bm25)} chunks")

    def _save_state(self):
        with sqlite3.connect(self.state_db) as conn:
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES('next_message_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.message_id),),
            )

    def _init_state_db(self):
        with sqlite3.connect(self.state_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    tombstoned INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(role, content_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def _migrate_json_state(self):
        if not self.state_file.exists():
            return

        with sqlite3.connect(self.state_db) as conn:
            existing = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if existing:
                return

            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            raw_log = data.get("raw_log", [])
            for index, message in enumerate(raw_log):
                text = message.get("text", "")
                role = message.get("role", "")
                message_id = message.get("message_id", message.get("turn_id", index))
                content_hash = message.get("content_hash") or self._content_hash(text)
                created_at = message.get("created_at") or self._now_iso()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO messages(
                        message_id, role, text, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (message_id, role, text, content_hash, created_at),
                )

            next_message_id = data.get("message_id", data.get("turn_id"))
            if next_message_id is None:
                next_message_id = max(
                    [message.get("message_id", message.get("turn_id", -1)) for message in raw_log],
                    default=-1,
                ) + 1
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES('next_message_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(next_message_id),),
            )

    def _persist_message(self, message: dict):
        with sqlite3.connect(self.state_db) as conn:
            conn.execute(
                """
                INSERT INTO messages(message_id, role, text, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message["message_id"],
                    message["role"],
                    message["text"],
                    message["content_hash"],
                    message["created_at"],
                ),
            )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _content_hash(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_message(self, role: str, text: str, extract_structured: bool = True) -> AddMessageResult:
        content_hash = self._content_hash(text)
        dedup_key = (role, content_hash)
        if dedup_key in self._message_hashes:
            message_id = self._message_hashes[dedup_key]
            print(f"  [dedup skipped] message_id={message_id}")
            return AddMessageResult(
                saved=False,
                deduped=True,
                message_id=message_id,
                content_hash=content_hash,
                chunk_ids=[],
                structured_object_ids=[],
                queued_job_ids=[],
            )

        message_id = self.message_id
        self.raw_log.append({
            "role": role,
            "text": text,
            "message_id": message_id,
            "content_hash": content_hash,
            "created_at": self._now_iso(),
        })
        self._message_hashes[dedup_key] = message_id
        chunks = self.chunker.split(text, self.message_id, role)
        if not chunks:
            self._persist_message(self.raw_log[-1])
            self.message_id += 1
            self._save_state()
            return AddMessageResult(
                saved=True,
                deduped=False,
                message_id=message_id,
                content_hash=content_hash,
                chunk_ids=[],
                structured_object_ids=[],
                queued_job_ids=[],
            )

        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"message_id": c.message_id, "role": c.role, "importance": c.importance}
                for c in chunks
            ],
        )
        for c in chunks:
            self.bm25.add(c.id, c.text)

        if extract_structured:
            structured = self.artifact_extractor.extract(role, text, self.message_id)
            exact_sources = {obj.source_text.strip() for obj in structured}
            llm_structured = self.structured_extractor.extract(role, text, self.message_id)
            structured.extend(
                obj for obj in llm_structured
                if obj.source_text.strip() not in exact_sources
                and not self._duplicates_exact_artifact(obj, exact_sources)
            )
            self.structured.add_many(structured)
            if structured:
                print(f"  [structured {len(structured)} object(s) | total: {len(self.structured)}]")
        else:
            structured = []

        print(f"  [stored {len(chunks)} chunk(s) | total: {self.collection.count()}]")
        self._persist_message(self.raw_log[-1])
        self.message_id += 1
        self._save_state()
        return AddMessageResult(
            saved=True,
            deduped=False,
            message_id=message_id,
            content_hash=content_hash,
            chunk_ids=[c.id for c in chunks],
            structured_object_ids=[obj.id for obj in structured],
            queued_job_ids=[],
        )

    # ── Retrieval (hybrid BM25 + embeddings via RRF) ──────────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVE_TOP_K) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0:
            return []

        fetch_k = min(top_k * 2, count)

        # embedding retrieval
        emb_results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "metadatas"],
        )
        emb_ids = emb_results["ids"][0]
        emb_rank = {id: rank for rank, id in enumerate(emb_ids)}

        # BM25 retrieval
        bm25_ids = self.bm25.search(query, top_k=fetch_k)
        bm25_rank = {id: rank for rank, id in enumerate(bm25_ids)}

        # RRF merge
        all_ids = list(set(emb_ids) | set(bm25_ids))
        rrf_scores = {
            id: (1 / (RRF_K + emb_rank[id] + 1) if id in emb_rank else 0)
               + (1 / (RRF_K + bm25_rank[id] + 1) if id in bm25_rank else 0)
            for id in all_ids
        }
        merged_ids = sorted(all_ids, key=lambda x: -rrf_scores[x])[:fetch_k]

        # fetch full docs for merged IDs
        fetched = self.collection.get(ids=merged_ids, include=["documents", "metadatas"])
        id_to_chunk = {
            id: RetrievedChunk(
                id=id,
                text=doc,
                importance=meta.get("importance", 0.5),
                message_id=meta.get("message_id", meta.get("turn_id", 0)),
                score=rrf_scores.get(id, 0.0),
            )
            for id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
        }

        seen, unique = set(), []
        for id in merged_ids:
            chunk = id_to_chunk.get(id)
            if chunk and chunk.text not in seen:
                seen.add(chunk.text)
                unique.append(chunk)
            if len(unique) == top_k:
                break
        return unique

    def retrieve_structured(
        self, query: str, top_k: int = STRUCTURED_TOP_K
    ) -> list[StructuredMemoryObject]:
        return self.structured.search(query, top_k=top_k)

    # ── Context builder ───────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def build_context(self, user_message: str) -> str:
        recent = self.raw_log[-RECENT_MESSAGES:] if RECENT_MESSAGES > 0 else []
        recent_texts = {m["text"] for m in recent}

        structured = self.retrieve_structured(user_message)
        retrieved = [r for r in self.retrieve(user_message) if r.text not in recent_texts]

        # ledger recovery
        if self.ledger.looks_like_missing_context(user_message) and len(self.ledger) > 0:
            for entry in self.ledger.search(user_message, top_k=2):
                if entry.text not in recent_texts:
                    retrieved.append(RetrievedChunk(
                        id=entry.chunk_id, text=entry.text,
                        importance=entry.importance, message_id=entry.message_id,
                        score=entry.importance,
                    ))
            print(f"  [ledger recovery] searched archived chunks")

        # compress to budget
        budget = CONTEXT_TOKEN_BUDGET
        kept, dropped = [], []
        for chunk in sorted(retrieved, key=lambda c: (-c.score, -c.message_id)):
            tokens = self._estimate_tokens(chunk.text)
            if budget - tokens >= 0:
                kept.append(chunk)
                budget -= tokens
            else:
                dropped.append(chunk)

        for chunk in dropped:
            self.ledger.log(chunk, reason="budget")

        if dropped:
            print(f"  [compression] kept {len(kept)}, dropped {len(dropped)} to ledger")

        parts = []
        if structured:
            parts.append("=== Structured Memory ===\n" + "\n---\n".join(
                self._format_structured(obj) for obj in structured
            ))
        if kept:
            parts.append("=== Relevant Memory ===\n" + "\n---\n".join(c.text for c in kept))
        if recent:
            lines = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in recent)
            parts.append("=== Recent Conversation ===\n" + lines)

        return "\n\n".join(parts)

    def _format_structured(self, obj: StructuredMemoryObject) -> str:
        tags = ", ".join(obj.tags)
        return (
            f"[{obj.type} | importance={obj.importance} | message_id={obj.message_id}]\n"
            f"Summary: {obj.summary}\n"
            f"Tags: {tags}\n"
            f"Source: {obj.source_text}"
        )

    def _duplicates_exact_artifact(
        self, obj: StructuredMemoryObject, exact_sources: set[str]
    ) -> bool:
        if obj.type not in EXACT_ARTIFACT_TYPES:
            return False
        source = obj.source_text.strip()
        return any(source == exact or exact in source or source in exact for exact in exact_sources)
