"""
Run a small retrieval benchmark from a labeled JSON corpus.

Run:
    uv run python scripts/benchmark_retrieval.py
    uv run python scripts/benchmark_retrieval.py --embedding-provider chroma_default
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "tests" / "golden" / "retrieval_cases.json"
DEFAULT_BENCH_DIR = ROOT / ".data" / "bench_retrieval"


@dataclass
class BenchHit:
    text: str
    source: str


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _set_embedding_env(args: argparse.Namespace) -> str:
    provider = args.embedding_provider.strip()
    model = args.embedding_model.strip()
    device = args.embedding_device.strip()
    normalize = "true" if args.normalize_embeddings else "false"

    os.environ["RAGMEMORY_EMBEDDING_PROVIDER"] = provider
    if model:
        os.environ["RAGMEMORY_EMBEDDING_MODEL"] = model
    else:
        os.environ.pop("RAGMEMORY_EMBEDDING_MODEL", None)
    os.environ["RAGMEMORY_EMBEDDING_DEVICE"] = device
    os.environ["RAGMEMORY_EMBEDDING_NORMALIZE"] = normalize

    return provider if not model else f"{provider}:{model}"


def _load_corpus(path: Path) -> dict:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corpus.get("cases"), list):
        raise ValueError(f"{path} must contain a cases list")
    return corpus


def _first_match_rank(results, expected: list[str]) -> int | None:
    found_ranks = []
    for expected_text in expected:
        found = None
        for index, result in enumerate(results, start=1):
            if expected_text in result.text:
                found = index
                break
        if found is None:
            return None
        found_ranks.append(found)
    return max(found_ranks) if found_ranks else None


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _copy_source_db(source_db: Path, bench_db: Path) -> None:
    bench_db.mkdir(parents=True, exist_ok=True)
    for name in ("state.sqlite", "structured_memory.jsonl", "ledger.json"):
        src = source_db / name
        if src.exists():
            shutil.copy2(src, bench_db / name)


def _query_hits(store, query: str, top_k: int, mode: str) -> list[BenchHit]:
    if mode == "search":
        return [
            BenchHit(text=result.text, source=result.source)
            for result in store.search(query, top_k=top_k)
        ]

    store.configure_recall(
        retrieve_top_k=top_k,
        structured_top_k=top_k,
        recent_messages=0,
        context_token_budget=100_000,
        include_recent=False,
        include_structured=True,
    )
    bundle = store.build_context_bundle(query)
    hits: list[BenchHit] = []
    seen: set[str] = set()
    for obj in bundle.structured:
        text = f"{obj.summary}\n{obj.source_text}"
        if text not in seen:
            seen.add(text)
            hits.append(BenchHit(text=text, source=f"structured:{obj.type}"))
    for chunk in bundle.retrieved + bundle.ledger_recovered:
        if chunk.text not in seen:
            seen.add(chunk.text)
            hits.append(BenchHit(text=chunk.text, source=chunk.source))
    return hits[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RagMemory retrieval on labeled golden cases."
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="JSON file with messages and cases.",
    )
    parser.add_argument(
        "--bench-db",
        default="",
        help="Benchmark DB path. Defaults to .data/bench_retrieval/<model-slug>.",
    )
    parser.add_argument(
        "--source-db",
        default="",
        help="Existing RagMemory DB to copy/reindex instead of seeding from cases messages.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=os.environ.get("RAGMEMORY_EMBEDDING_PROVIDER", "sentence_transformers"),
        help="Embedding provider: chroma_default or sentence_transformers.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("RAGMEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        help="Embedding model for sentence_transformers.",
    )
    parser.add_argument(
        "--embedding-device",
        default=os.environ.get("RAGMEMORY_EMBEDDING_DEVICE", "cpu"),
        help="Embedding device.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize_embeddings",
        help="Disable sentence-transformers embedding normalization.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum retrieval depth used for metrics.",
    )
    parser.add_argument(
        "--mode",
        choices=("context", "search"),
        default="context",
        help="context benchmarks recall bundle output; search benchmarks chunk search only.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the generated benchmark DB instead of deleting/rebuilding it.",
    )
    args = parser.parse_args()

    if args.max_k <= 0:
        raise ValueError("--max-k must be positive")

    label = _set_embedding_env(args)
    bench_db = Path(args.bench_db) if args.bench_db else DEFAULT_BENCH_DIR / _slug(label)
    cases_path = Path(args.cases)

    if bench_db.exists() and not args.keep_db:
        shutil.rmtree(bench_db)

    # Import after embedding env is set; ragmemory loads local config at import time.
    import ragmemory.memory as memory
    from ragmemory.memory import MemoryStore

    original_recent = memory.RECENT_MESSAGES
    memory.RECENT_MESSAGES = 0

    try:
        started = time.perf_counter()
        corpus = _load_corpus(cases_path)
        source_db = Path(args.source_db) if args.source_db else None
        if source_db:
            if not (source_db / "state.sqlite").exists():
                raise FileNotFoundError(f"{source_db / 'state.sqlite'} not found")
            if bench_db.exists() and not args.keep_db:
                shutil.rmtree(bench_db)
            if not bench_db.exists():
                _copy_source_db(source_db, bench_db)
            store = MemoryStore(db_path=str(bench_db))
            if not args.keep_db:
                store.rebuild_chat_memory_index()
                store.rebuild_structured_memory_index()
        else:
            messages = corpus.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{cases_path} must contain messages unless --source-db is set")
            store = MemoryStore(db_path=str(bench_db))
            for message in messages:
                store.add_message(
                    message["role"],
                    message["text"],
                    extract_structured=False,
                )
        build_seconds = time.perf_counter() - started

        rows = []
        latencies_ms = []
        for case in corpus["cases"]:
            top_k = min(max(args.max_k, case.get("top_k", 0)), args.max_k)
            query_started = time.perf_counter()
            results = _query_hits(store, case["query"], top_k=top_k, mode=args.mode)
            latency_ms = (time.perf_counter() - query_started) * 1000
            latencies_ms.append(latency_ms)

            expected = case.get("must_contain", [])
            rank = _first_match_rank(results, expected)
            rows.append(
                {
                    "name": case.get("name", case["query"]),
                    "rank": rank,
                    "latency_ms": latency_ms,
                    "expected": expected,
                    "top_result": results[0].text if results else "",
                    "top_source": results[0].source if results else "",
                }
            )

        total = len(rows)
        recall_at_5 = sum(1 for row in rows if row["rank"] and row["rank"] <= 5) / total
        recall_at_10 = sum(1 for row in rows if row["rank"] and row["rank"] <= 10) / total
        mrr = sum((1 / row["rank"]) if row["rank"] else 0 for row in rows) / total
        p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
        p95 = sorted(latencies_ms)[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95))] if latencies_ms else 0.0

        print(f"Cases: {cases_path}")
        print(f"Embedding: {store.embedding_options.label}")
        print(f"Mode: {args.mode}")
        print(f"DB: {bench_db}")
        print(f"Build/index time: {build_seconds:.2f}s")
        print()
        print("case                           rank  latency_ms")
        print("-----------------------------  ----  ----------")
        for row in rows:
            rank = str(row["rank"]) if row["rank"] else "miss"
            print(f"{row['name'][:29]:29}  {rank:>4}  {row['latency_ms']:10.1f}")
        print()
        print(
            "summary: "
            f"recall@5={_percent(recall_at_5)} "
            f"recall@10={_percent(recall_at_10)} "
            f"mrr={mrr:.3f} "
            f"p50_ms={p50:.1f} "
            f"p95_ms={p95:.1f}"
        )
    finally:
        memory.RECENT_MESSAGES = original_recent


if __name__ == "__main__":
    main()
