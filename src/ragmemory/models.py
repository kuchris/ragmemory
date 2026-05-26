from __future__ import annotations

import hashlib
from dataclasses import dataclass

EVIDENCE_REF_HASH_CHARS = 12


def evidence_content_hash(text: str, ev_type: str) -> str:
    # Schema v1: normalize line endings and trim outer whitespace.
    # Do not lowercase block/code evidence; exact casing is part of the evidence.
    # Keep this stable, because compact_text evidence refs depend on it.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if ev_type in {"path", "url"}:
        normalized = normalized.lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:EVIDENCE_REF_HASH_CHARS]


def score_importance(text: str) -> float:
    """Compatibility hook for old callers; raw chunk retention is score-based."""
    return 0.5


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
    source: str = "raw_chunk"


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
    content_hash: str | None = None


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
class MessageCompactionJob:
    job_id: str
    role: str
    text: str
    message_id: int


@dataclass
class EvidenceReference:
    ref_type: str
    content_hash: str
    source_text: str
    preview: str

    @property
    def marker(self) -> str:
        return f"evidence[{self.ref_type}:{self.content_hash}]"


@dataclass
class BackgroundJob:
    job_id: str
    job_type: str
    message_id: int
    attempts: int


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
    recent_token_budget_ratio: float = 0.4
