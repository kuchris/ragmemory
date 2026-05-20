import os
import re
import uuid
import json
import hashlib
import sqlite3
import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def _load_settings_file(path: Path) -> None:
    if not path.exists():
        return
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8-sig")
    if parser.has_section("structured_memory"):
        section = parser["structured_memory"]
        if section.get("api_key"):
            os.environ.setdefault("NVIDIA_API_KEY", section["api_key"].strip())
        if section.get("model"):
            os.environ.setdefault("STRUCTURED_MEMORY_MODEL", section["model"].strip())


def _load_local_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    seen = set()
    for path in (
        repo_root / "ragmemory.local.ini",
        Path.cwd() / "ragmemory.local.ini",
        Path.home() / ".ragmemory.ini",
        repo_root / ".env",
        Path.cwd() / ".env",
        Path.home() / ".ragmemory.env",
    ):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.suffix == ".ini":
            _load_settings_file(resolved)
        else:
            _load_env_file(resolved)


_load_local_config()

CHUNK_MAX_TOKENS = 300
CHUNK_MIN_TOKENS = 80
RECENT_MESSAGES = 12
RETRIEVE_TOP_K = 5
STRUCTURED_TOP_K = 3
CONTEXT_TOKEN_BUDGET = 2000
RRF_K = 60  # reciprocal rank fusion constant
DECAY_FETCH_MIN = 50
DECAY_SCORE_FLOOR = 0.3
DEFAULT_HALF_LIFE_DAYS = {
    "raw_message": 7.0,
    "debug_log": 1.0,
    "open_question": 14.0,
    "config": 30.0,
    "code_reference": 30.0,
    "decision": 90.0,
    "constraint": 90.0,
    "preference": 180.0,
    "identity": 365.0,
}
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
STRUCTURED_MEMORY_MODEL = os.environ.get(
    "STRUCTURED_MEMORY_MODEL", "minimaxai/minimax-m2.7"
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
    decay_strength: float = 1.0
    rrf_score: float = 0.0


@dataclass
class MemoryMetadata:
    message_id: int
    memory_type: str
    created_at: str
    last_accessed_at: str
    access_count: int
    base_importance: float
    half_life_days: float
    pinned: bool
    superseded_by: str | None
    tombstoned_at: str | None


@dataclass
class SearchResult:
    item_id: str
    item_type: str
    text: str
    score: float
    message_id: int | None
    source: str
    metadata: dict


@dataclass
class MessageRecord:
    role: str
    text: str
    message_id: int
    content_hash: str
    created_at: str | None = None


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


@dataclass
class StructuredExtractionJob:
    job_id: str
    role: str
    text: str
    message_id: int


@dataclass
class ForgetPreview:
    messages: list[MessageRecord]
    chunks: list[RetrievedChunk]
    structured: list[StructuredMemoryObject]
    ledger_entries: list[LedgerEntry]
    message_count: int
    chunk_count: int
    structured_count: int
    ledger_count: int
    truncated: bool


@dataclass
class ForgetResult(ForgetPreview):
    tombstoned_count: int
    event_id: str


@dataclass
class ContextBundle:
    query: str
    recent: list[MessageRecord]
    structured: list[StructuredMemoryObject]
    retrieved: list[RetrievedChunk]
    ledger_recovered: list[RetrievedChunk]
    kept: list[RetrievedChunk]
    would_be_dropped: list[RetrievedChunk]
    token_budget: int
    tokens_used: int


@dataclass
class RecallOptions:
    retrieve_top_k: int | None = None
    structured_top_k: int | None = None
    context_token_budget: int | None = None
    recent_messages: int | None = None
    include_recent: bool = True
    include_structured: bool = True


def format_for_prompt(bundle: ContextBundle) -> str:
    parts = []
    if bundle.structured:
        parts.append("=== Structured Memory ===\n" + "\n---\n".join(
            _format_structured_object(obj) for obj in bundle.structured
        ))
    if bundle.kept:
        parts.append("=== Relevant Memory ===\n" + "\n---\n".join(
            chunk.text for chunk in bundle.kept
        ))
    if bundle.recent:
        lines = "\n".join(f"{message.role.upper()}: {message.text}" for message in bundle.recent)
        parts.append("=== Recent Conversation ===\n" + lines)
    return "\n\n".join(parts)


def _format_structured_object(obj: StructuredMemoryObject) -> str:
    tags = ", ".join(obj.tags)
    return (
        f"[{obj.type} | importance={obj.importance} | message_id={obj.message_id}]\n"
        f"Summary: {obj.summary}\n"
        f"Tags: {tags}\n"
        f"Source: {obj.source_text}"
    )


# ── BM25 index ────────────────────────────────────────────────────────────────

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
        self.events_file = self.db_path / "events.jsonl"

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
        self._tombstoned_message_ids: set[int] = set()
        self._pending_extractions: list[StructuredExtractionJob] = []
        self._recall_options = RecallOptions()
        self.message_id = 0

        self._load_state()

    def configure_recall(
        self,
        *,
        retrieve_top_k: int | None = None,
        structured_top_k: int | None = None,
        context_token_budget: int | None = None,
        recent_messages: int | None = None,
        include_recent: bool = True,
        include_structured: bool = True,
    ) -> None:
        self._recall_options = RecallOptions(
            retrieve_top_k=max(0, retrieve_top_k) if retrieve_top_k is not None else None,
            structured_top_k=max(0, structured_top_k) if structured_top_k is not None else None,
            context_token_budget=(
                max(0, context_token_budget)
                if context_token_budget is not None else None
            ),
            recent_messages=max(0, recent_messages) if recent_messages is not None else None,
            include_recent=include_recent,
            include_structured=include_structured,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self):
        self._init_state_db()
        self._migrate_json_state()
        with sqlite3.connect(self.state_db) as conn:
            self._backfill_memory_metadata(conn)
        with sqlite3.connect(self.state_db) as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, text, content_hash, created_at
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
                    "created_at": created_at,
                }
                for message_id, role, text, content_hash, created_at in rows
            ]
            next_id = conn.execute(
                "SELECT value FROM meta WHERE key = 'next_message_id'"
            ).fetchone()
            tombstoned_rows = conn.execute(
                "SELECT message_id FROM messages WHERE tombstoned = 1"
            ).fetchall()

        for message in self.raw_log:
            self._message_hashes[(message["role"], message["content_hash"])] = message["message_id"]
        self._tombstoned_message_ids = {message_id for (message_id,) in tombstoned_rows}

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

    def _sync_message_state_from_db(self):
        with sqlite3.connect(self.state_db) as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, text, content_hash, created_at
                FROM messages
                WHERE tombstoned = 0
                ORDER BY message_id
                """
            ).fetchall()
            tombstoned_rows = conn.execute(
                "SELECT message_id FROM messages WHERE tombstoned = 1"
            ).fetchall()
            next_id = conn.execute(
                "SELECT value FROM meta WHERE key = 'next_message_id'"
            ).fetchone()
            max_id = conn.execute("SELECT MAX(message_id) FROM messages").fetchone()[0]

        self.raw_log = [
            {
                "role": role,
                "text": text,
                "message_id": message_id,
                "content_hash": content_hash,
                "created_at": created_at,
            }
            for message_id, role, text, content_hash, created_at in rows
        ]
        self._message_hashes = {
            (message["role"], message["content_hash"]): message["message_id"]
            for message in self.raw_log
        }
        self._tombstoned_message_ids = {message_id for (message_id,) in tombstoned_rows}
        db_next = int(next_id[0]) if next_id else 0
        max_next = (max_id + 1) if max_id is not None else 0
        self.message_id = max(self.message_id, db_next, max_next)

    def _allocate_message_id(self) -> int:
        with sqlite3.connect(self.state_db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            next_id_row = conn.execute(
                "SELECT value FROM meta WHERE key = 'next_message_id'"
            ).fetchone()
            max_id = conn.execute("SELECT MAX(message_id) FROM messages").fetchone()[0]
            db_next = int(next_id_row[0]) if next_id_row else 0
            max_next = (max_id + 1) if max_id is not None else 0
            message_id = max(self.message_id, db_next, max_next)
            next_message_id = message_id + 1
            conn.execute(
                """
                INSERT INTO meta(key, value) VALUES('next_message_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(next_message_id),),
            )
            self.message_id = next_message_id
            return message_id

    def _init_state_db(self):
        with sqlite3.connect(self.state_db) as conn:
            self._create_messages_table(conn)
            self._create_memory_metadata_table(conn)
            self._migrate_message_uniqueness(conn)
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS messages_active_hash
                ON messages(role, content_hash)
                WHERE tombstoned = 0
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
            self._backfill_memory_metadata(conn)

    def _create_messages_table(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tombstoned INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    def _create_memory_metadata_table(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_metadata (
                message_id INTEGER PRIMARY KEY,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                base_importance REAL NOT NULL DEFAULT 1.0,
                half_life_days REAL NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT,
                tombstoned_at TEXT,
                FOREIGN KEY(message_id) REFERENCES messages(message_id)
            )
            """
        )

    def _migrate_message_uniqueness(self, conn):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
        ).fetchone()
        table_sql = row[0] if row else ""
        if "UNIQUE(role,content_hash)" not in table_sql.replace(" ", ""):
            return

        conn.execute("ALTER TABLE messages RENAME TO messages_old_unique")
        self._create_messages_table(conn)
        conn.execute(
            """
            INSERT INTO messages(
                message_id, role, text, content_hash, created_at, tombstoned
            )
            SELECT message_id, role, text, content_hash, created_at, tombstoned
            FROM messages_old_unique
            """
        )
        conn.execute("DROP TABLE messages_old_unique")

    def _backfill_memory_metadata(self, conn):
        default_half_life = DEFAULT_HALF_LIFE_DAYS["raw_message"]
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_metadata(
                message_id, memory_type, created_at, last_accessed_at,
                access_count, base_importance, half_life_days, pinned,
                tombstoned_at
            )
            SELECT
                message_id,
                'raw_message',
                created_at,
                created_at,
                0,
                1.0,
                ?,
                0,
                CASE WHEN tombstoned = 1 THEN created_at ELSE NULL END
            FROM messages
            """,
            (default_half_life,),
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
            self._persist_memory_metadata(conn, message)

    def _persist_memory_metadata(self, conn, message: dict):
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_metadata(
                message_id, memory_type, created_at, last_accessed_at,
                access_count, base_importance, half_life_days, pinned
            ) VALUES (?, ?, ?, ?, 0, 1.0, ?, 0)
            """,
            (
                message["message_id"],
                "raw_message",
                message["created_at"],
                message["created_at"],
                DEFAULT_HALF_LIFE_DAYS["raw_message"],
            ),
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log_event(self, event: str, **payload):
        event_id = str(uuid.uuid4())
        record = {
            "event_id": event_id,
            "ts": self._now_iso(),
            "event": event,
            **payload,
        }
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return event_id

    def _content_hash(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_message_tombstoned(self, message_id: int) -> bool:
        return message_id in self._tombstoned_message_ids

    def _parse_iso_datetime(self, value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _load_memory_metadata(self, message_ids: set[int]) -> dict[int, MemoryMetadata]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        with sqlite3.connect(self.state_db) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    message_id, memory_type, created_at, last_accessed_at,
                    access_count, base_importance, half_life_days, pinned,
                    superseded_by, tombstoned_at
                FROM memory_metadata
                WHERE message_id IN ({placeholders})
                """,
                tuple(message_ids),
            ).fetchall()
        return {
            message_id: MemoryMetadata(
                message_id=message_id,
                memory_type=memory_type,
                created_at=created_at,
                last_accessed_at=last_accessed_at,
                access_count=access_count,
                base_importance=base_importance,
                half_life_days=half_life_days,
                pinned=bool(pinned),
                superseded_by=superseded_by,
                tombstoned_at=tombstoned_at,
            )
            for (
                message_id,
                memory_type,
                created_at,
                last_accessed_at,
                access_count,
                base_importance,
                half_life_days,
                pinned,
                superseded_by,
                tombstoned_at,
            ) in rows
        }

    def _decay_strength(self, metadata: MemoryMetadata | None, now: datetime) -> float:
        if metadata is None or metadata.pinned:
            return 1.0
        half_life_days = max(metadata.half_life_days, 0.001)
        last_accessed = self._parse_iso_datetime(metadata.last_accessed_at)
        age_days = max((now - last_accessed).total_seconds() / 86400, 0.0)
        strength = metadata.base_importance * (2 ** (-age_days / half_life_days))
        return min(max(strength, 0.0), 1.0)

    def _is_metadata_active(self, metadata: MemoryMetadata | None) -> bool:
        return metadata is None or (
            metadata.tombstoned_at is None and metadata.superseded_by is None
        )

    def _touch_memory_metadata(self, message_ids: set[int]) -> None:
        active_ids = [
            message_id for message_id in sorted(message_ids)
            if not self._is_message_tombstoned(message_id)
        ]
        if not active_ids:
            return
        now = self._now_iso()
        with sqlite3.connect(self.state_db) as conn:
            conn.executemany(
                """
                UPDATE memory_metadata
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE message_id = ?
                  AND tombstoned_at IS NULL
                  AND superseded_by IS NULL
                """,
                [(now, message_id) for message_id in active_ids],
            )
        self._log_event(
            "memory_accessed",
            message_ids=active_ids,
            count=len(active_ids),
            reason="context_injected",
            citation_evidence=False,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_message(
        self, role: str, text: str, extract_structured: bool | str = True
    ) -> AddMessageResult:
        self._sync_message_state_from_db()
        content_hash = self._content_hash(text)
        dedup_key = (role, content_hash)
        if dedup_key in self._message_hashes:
            message_id = self._message_hashes[dedup_key]
            print(f"  [dedup skipped] message_id={message_id}")
            self._log_event(
                "message_deduped",
                role=role,
                message_id=message_id,
                content_hash=content_hash,
            )
            return AddMessageResult(
                saved=False,
                deduped=True,
                message_id=message_id,
                content_hash=content_hash,
                chunk_ids=[],
                structured_object_ids=[],
                queued_job_ids=[],
            )

        message_id = self._allocate_message_id()
        self.raw_log.append({
            "role": role,
            "text": text,
            "message_id": message_id,
            "content_hash": content_hash,
            "created_at": self._now_iso(),
        })
        self._message_hashes[dedup_key] = message_id
        chunks = self.chunker.split(text, message_id, role)
        if not chunks:
            self._persist_message(self.raw_log[-1])
            self._log_event(
                "message_saved",
                role=role,
                message_id=message_id,
                content_hash=content_hash,
                chunk_ids=[],
                structured_object_ids=[],
            )
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

        queued_job_ids = []
        if extract_structured == "background":
            structured = []
            queued_job_ids = [self._queue_structured_extraction(role, text, message_id)]
        elif extract_structured:
            structured = self._extract_structured_objects(role, text, message_id)
            self.structured.add_many(structured)
            self._log_structured_objects_added(structured, reason="message_saved")
            if structured:
                print(f"  [structured {len(structured)} object(s) | total: {len(self.structured)}]")
        else:
            structured = []

        print(f"  [stored {len(chunks)} chunk(s) | total: {self.collection.count()}]")
        self._persist_message(self.raw_log[-1])
        self._log_event(
            "message_saved",
            role=role,
            message_id=message_id,
            content_hash=content_hash,
            chunk_ids=[c.id for c in chunks],
            structured_object_ids=[obj.id for obj in structured],
            queued_job_ids=queued_job_ids,
        )
        return AddMessageResult(
            saved=True,
            deduped=False,
            message_id=message_id,
            content_hash=content_hash,
            chunk_ids=[c.id for c in chunks],
            structured_object_ids=[obj.id for obj in structured],
            queued_job_ids=queued_job_ids,
        )

    def _extract_structured_objects(
        self, role: str, text: str, message_id: int
    ) -> list[StructuredMemoryObject]:
        structured = self.artifact_extractor.extract(role, text, message_id)
        exact_sources = {obj.source_text.strip() for obj in structured}
        llm_structured = self.structured_extractor.extract(role, text, message_id)
        structured.extend(
            obj for obj in llm_structured
            if obj.source_text.strip() not in exact_sources
            and not self._duplicates_exact_artifact(obj, exact_sources)
        )
        return structured

    def _log_structured_objects_added(
        self, objects: list[StructuredMemoryObject], reason: str, job_id: str | None = None
    ) -> None:
        for obj in objects:
            payload = {
                "structured_object_id": obj.id,
                "type": obj.type,
                "tags": obj.tags,
                "message_id": obj.message_id,
                "role": obj.role,
                "importance": obj.importance,
                "reason": reason,
            }
            if job_id:
                payload["job_id"] = job_id
            self._log_event("structured_object_added", **payload)

    def _queue_structured_extraction(self, role: str, text: str, message_id: int) -> str:
        job = StructuredExtractionJob(
            job_id=f"structured_{uuid.uuid4()}",
            role=role,
            text=text,
            message_id=message_id,
        )
        self._pending_extractions.append(job)
        self._log_event(
            "structured_extraction_queued",
            job_id=job.job_id,
            message_id=message_id,
        )
        return job.job_id

    def run_pending_extractions(self, limit: int | None = None) -> list[str]:
        if limit is not None and limit <= 0:
            return []
        max_jobs = len(self._pending_extractions) if limit is None else limit
        remaining = []
        structured_object_ids = []

        for job in self._pending_extractions:
            if max_jobs <= 0:
                remaining.append(job)
                continue
            max_jobs -= 1
            if self._is_message_tombstoned(job.message_id):
                self._log_event(
                    "structured_extraction_skipped",
                    job_id=job.job_id,
                    message_id=job.message_id,
                    reason="message_tombstoned",
                )
                continue
            structured = self._extract_structured_objects(
                job.role, job.text, job.message_id
            )
            self.structured.add_many(structured)
            self._log_structured_objects_added(
                structured,
                reason="structured_extraction_completed",
                job_id=job.job_id,
            )
            object_ids = [obj.id for obj in structured]
            structured_object_ids.extend(object_ids)
            self._log_event(
                "structured_extraction_completed",
                job_id=job.job_id,
                message_id=job.message_id,
                structured_object_ids=object_ids,
            )
            if structured:
                print(f"  [structured {len(structured)} object(s) | total: {len(self.structured)}]")

        self._pending_extractions = remaining
        return structured_object_ids

    # ── Retrieval (hybrid BM25 + embeddings via RRF) ──────────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVE_TOP_K) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0 or top_k <= 0:
            return []

        fetch_k = min(max(top_k * 10, DECAY_FETCH_MIN), count)
        if self._tombstoned_message_ids:
            fetch_k = count

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
        message_ids = {
            (meta or {}).get("message_id", (meta or {}).get("turn_id", 0))
            for meta in fetched["metadatas"]
        }
        memory_metadata = self._load_memory_metadata(message_ids)
        now = datetime.now(timezone.utc)
        id_to_chunk = {
            id: RetrievedChunk(
                id=id,
                text=doc,
                importance=(meta or {}).get("importance", 0.5),
                message_id=(meta or {}).get("message_id", (meta or {}).get("turn_id", 0)),
                score=rrf_scores.get(id, 0.0),
                rrf_score=rrf_scores.get(id, 0.0),
            )
            for id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
        }
        for chunk in id_to_chunk.values():
            metadata = memory_metadata.get(chunk.message_id)
            strength = self._decay_strength(metadata, now)
            chunk.decay_strength = strength
            chunk.score = chunk.rrf_score * (
                DECAY_SCORE_FLOOR + (1 - DECAY_SCORE_FLOOR) * strength
            )

        ranked_ids = sorted(
            (id for id in merged_ids if id in id_to_chunk),
            key=lambda id: -id_to_chunk[id].score,
        )
        seen, unique = set(), []
        for id in ranked_ids:
            chunk = id_to_chunk.get(id)
            metadata = memory_metadata.get(chunk.message_id) if chunk else None
            if (
                chunk
                and not self._is_message_tombstoned(chunk.message_id)
                and self._is_metadata_active(metadata)
                and chunk.text not in seen
            ):
                seen.add(chunk.text)
                unique.append(chunk)
            if len(unique) == top_k:
                break
        self._log_event(
            "chunks_retrieved",
            query=query,
            result_count=len(unique),
            chunk_ids=[chunk.id for chunk in unique],
            scores=[round(chunk.score, 6) for chunk in unique],
            rrf_scores=[round(chunk.rrf_score, 6) for chunk in unique],
            decay_strengths=[round(chunk.decay_strength, 6) for chunk in unique],
            message_ids=[chunk.message_id for chunk in unique],
        )
        return unique

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        limit = RETRIEVE_TOP_K if top_k is None else top_k
        if limit <= 0:
            return []
        return [
            SearchResult(
                item_id=chunk.id,
                item_type="chunk",
                text=chunk.text,
                score=chunk.score,
                message_id=chunk.message_id,
                source="raw_chunk",
                metadata={"importance": chunk.importance},
            )
            for chunk in self.retrieve(query, top_k=limit)
        ]

    def forget(
        self,
        *,
        message_ids: list[int] | None = None,
        before: str | datetime | None = None,
        query: str | None = None,
        confirm: bool = False,
        sample_limit: int = 50,
    ) -> ForgetPreview | ForgetResult:
        if query is not None:
            raise NotImplementedError("forget(query=...) is not implemented yet")
        if message_ids and before is not None:
            raise ValueError("forget accepts one selector at a time for now")

        selected_ids = self._select_forget_message_ids(message_ids, before)
        preview = self._build_forget_preview(selected_ids, sample_limit)
        if not confirm:
            return preview

        matched_messages = [
            message for message in self.raw_log
            if message["message_id"] in selected_ids
        ]
        matched_message_ids = [message["message_id"] for message in matched_messages]
        if matched_message_ids:
            now = self._now_iso()
            with sqlite3.connect(self.state_db) as conn:
                conn.executemany(
                    "UPDATE messages SET tombstoned = 1 WHERE message_id = ?",
                    [(message_id,) for message_id in matched_message_ids],
                )
                conn.executemany(
                    """
                    UPDATE memory_metadata
                    SET tombstoned_at = ?
                    WHERE message_id = ?
                    """,
                    [(now, message_id) for message_id in matched_message_ids],
                )
            self._tombstoned_message_ids.update(matched_message_ids)
            matched_id_set = set(matched_message_ids)
            self.raw_log = [
                message for message in self.raw_log
                if message["message_id"] not in matched_id_set
            ]
            for message in matched_messages:
                self._message_hashes.pop(
                    (message["role"], message["content_hash"]),
                    None,
                )

        event_id = self._log_event(
            "memory_tombstoned",
            message_ids=matched_message_ids,
            message_count=preview.message_count,
            chunk_count=preview.chunk_count,
            structured_count=preview.structured_count,
            ledger_count=preview.ledger_count,
        )
        return ForgetResult(
            messages=preview.messages,
            chunks=preview.chunks,
            structured=preview.structured,
            ledger_entries=preview.ledger_entries,
            message_count=preview.message_count,
            chunk_count=preview.chunk_count,
            structured_count=preview.structured_count,
            ledger_count=preview.ledger_count,
            truncated=preview.truncated,
            tombstoned_count=len(matched_message_ids),
            event_id=event_id,
        )

    def _select_forget_message_ids(
        self, message_ids: list[int] | None, before: str | datetime | None
    ) -> set[int]:
        if message_ids:
            return set(message_ids)
        if before is not None:
            cutoff = self._parse_forget_before(before)
            return {
                message["message_id"]
                for message in self.raw_log
                if self._parse_forget_before(message["created_at"]) < cutoff
            }
        raise ValueError("forget requires message_ids or before for now")

    def _parse_forget_before(self, value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _build_forget_preview(
        self, selected_ids: set[int], sample_limit: int
    ) -> ForgetPreview:
        messages = [
            MessageRecord(
                role=message["role"],
                text=message["text"],
                message_id=message["message_id"],
                content_hash=message["content_hash"],
                created_at=message.get("created_at"),
            )
            for message in self.raw_log
            if message["message_id"] in selected_ids
        ]
        chunks = self._chunks_for_message_ids(selected_ids)
        structured = [
            obj for obj in self.structured.objects.values()
            if obj.message_id in selected_ids
            and not self._is_message_tombstoned(obj.message_id)
        ]
        ledger_entries = [
            entry for entry in self.ledger.entries
            if entry.message_id in selected_ids
            and not self._is_message_tombstoned(entry.message_id)
        ]

        limit = max(sample_limit, 0)
        truncated = any(
            len(records) > limit
            for records in (messages, chunks, structured, ledger_entries)
        )
        return ForgetPreview(
            messages=messages[:limit],
            chunks=chunks[:limit],
            structured=structured[:limit],
            ledger_entries=ledger_entries[:limit],
            message_count=len(messages),
            chunk_count=len(chunks),
            structured_count=len(structured),
            ledger_count=len(ledger_entries),
            truncated=truncated,
        )

    def _chunks_for_message_ids(self, message_ids: set[int]) -> list[RetrievedChunk]:
        if not message_ids or self.collection.count() == 0:
            return []
        data = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            metadata = metadata or {}
            message_id = metadata.get("message_id", metadata.get("turn_id", 0))
            if message_id not in message_ids or self._is_message_tombstoned(message_id):
                continue
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    importance=metadata.get("importance", 0.5),
                    message_id=message_id,
                )
            )
        return chunks

    def retrieve_structured(
        self, query: str, top_k: int = STRUCTURED_TOP_K
    ) -> list[StructuredMemoryObject]:
        if top_k <= 0:
            return []
        fetch_k = len(self.structured) if self._tombstoned_message_ids else top_k
        return [
            obj for obj in self.structured.search(query, top_k=fetch_k)
            if not self._is_message_tombstoned(obj.message_id)
        ][:top_k]

    # ── Context builder ───────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def build_context_bundle(self, user_message: str) -> ContextBundle:
        options = self._recall_options
        recent_limit = (
            RECENT_MESSAGES if options.recent_messages is None else options.recent_messages
        )
        retrieve_top_k = (
            RETRIEVE_TOP_K if options.retrieve_top_k is None else options.retrieve_top_k
        )
        structured_top_k = (
            STRUCTURED_TOP_K
            if options.structured_top_k is None else options.structured_top_k
        )
        token_budget = (
            CONTEXT_TOKEN_BUDGET
            if options.context_token_budget is None else options.context_token_budget
        )
        recent = [
            MessageRecord(
                role=message["role"],
                text=message["text"],
                message_id=message["message_id"],
                content_hash=message["content_hash"],
                created_at=message.get("created_at"),
            )
            for message in (
                self.raw_log[-recent_limit:]
                if options.include_recent and recent_limit > 0 else []
            )
        ]
        recent_texts = {message.text for message in recent}

        structured = (
            self.retrieve_structured(user_message, top_k=structured_top_k)
            if options.include_structured else []
        )
        retrieved = [
            chunk for chunk in self.retrieve(user_message, top_k=retrieve_top_k)
            if chunk.text not in recent_texts
        ]

        ledger_recovered = []
        if self.ledger.looks_like_missing_context(user_message) and len(self.ledger) > 0:
            ledger_top_k = len(self.ledger.entries) if self._tombstoned_message_ids else 2
            for entry in self.ledger.search(user_message, top_k=ledger_top_k):
                if (
                    not self._is_message_tombstoned(entry.message_id)
                    and entry.text not in recent_texts
                ):
                    ledger_recovered.append(RetrievedChunk(
                        id=entry.chunk_id, text=entry.text,
                        importance=entry.importance, message_id=entry.message_id,
                        score=entry.importance,
                    ))
                if len(ledger_recovered) == 2:
                    break
            print(f"  [ledger recovery] searched archived chunks")

        budget = token_budget
        kept, would_be_dropped = [], []
        for chunk in sorted(retrieved + ledger_recovered, key=lambda c: (-c.score, -c.message_id)):
            tokens = self._estimate_tokens(chunk.text)
            if budget - tokens >= 0:
                kept.append(chunk)
                budget -= tokens
            else:
                would_be_dropped.append(chunk)

        bundle = ContextBundle(
            query=user_message,
            recent=recent,
            structured=structured,
            retrieved=retrieved,
            ledger_recovered=ledger_recovered,
            kept=kept,
            would_be_dropped=would_be_dropped,
            token_budget=token_budget,
            tokens_used=token_budget - budget,
        )
        self._touch_memory_metadata({chunk.message_id for chunk in kept})
        self._log_event(
            "context_built",
            query=user_message,
            structured_count=len(structured),
            retrieved_count=len(retrieved),
            ledger_recovered_count=len(ledger_recovered),
            kept_count=len(kept),
            dropped_count=len(would_be_dropped),
            recent_count=len(recent),
            token_budget=token_budget,
            retrieve_top_k=retrieve_top_k,
            structured_top_k=structured_top_k,
        )
        return bundle

    def commit_drops(self, bundle: ContextBundle):
        for chunk in bundle.would_be_dropped:
            self.ledger.log(chunk, reason="budget")

        if bundle.would_be_dropped:
            print(
                f"  [compression] kept {len(bundle.kept)}, "
                f"dropped {len(bundle.would_be_dropped)} to ledger"
            )
            self._log_event(
                "chunks_dropped_to_ledger",
                query=bundle.query,
                chunk_ids=[chunk.id for chunk in bundle.would_be_dropped],
                message_ids=[chunk.message_id for chunk in bundle.would_be_dropped],
                reason="budget",
            )

    def build_context(self, user_message: str) -> str:
        """Deprecated. Use build_context_bundle + format_for_prompt."""
        bundle = self.build_context_bundle(user_message)
        self.commit_drops(bundle)
        return format_for_prompt(bundle)

    def _format_structured(self, obj: StructuredMemoryObject) -> str:
        return _format_structured_object(obj)

    def _duplicates_exact_artifact(
        self, obj: StructuredMemoryObject, exact_sources: set[str]
    ) -> bool:
        if obj.type not in EXACT_ARTIFACT_TYPES:
            return False
        source = obj.source_text.strip()
        return any(source == exact or exact in source or source in exact for exact in exact_sources)
