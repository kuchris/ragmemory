"""
Build character-relationship graph edges from an ingested novel.

Reads all chunks from ChromaDB, calls LM Studio to extract character
names from each chunk, then writes "character" edges between chunks
that share the same character.

Run:
    uv run python build_character_edges.py
    uv run python build_character_edges.py --dry-run
    uv run python build_character_edges.py --top-k 5
    uv run python build_character_edges.py --db-path ./chroma_novel

Resume: interrupted runs are safe — chunk results are cached in
character_cache.json so already-processed chunks are skipped.
"""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
from pathlib import Path

import httpx

from memory import MemoryStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── LM Studio config (same as chat_novel.py) ──────────────────────────────
MODEL = os.environ.get("LMSTUDIO_MODEL", "gemma-4-e4b-uncensored-hauhaucs-aggressive")
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = "./chroma_novel"
GRAPH_EDGE_FILE = "graph_edges.jsonl"
CACHE_FILE = "character_cache.json"   # chunk_id -> [character, ...]
CHARACTER_NEIGHBORS = 5               # connect each chunk to N nearest per character
RETRY_DELAY = 2                       # seconds between retries on LLM failure

EXTRACT_PROMPT = """\
請從以下小說段落中，列出所有出現的人物角色名字。
只輸出角色名，以逗號分隔，不要解釋、不要標點以外的符號。
如果段落中沒有任何角色，只輸出：無

段落：
{text}"""


# ── LM Studio call ────────────────────────────────────────────────────────

def extract_characters_llm(text: str) -> list[str]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": EXTRACT_PROMPT.format(text=text[:1200])},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
        "stream": False,
    }
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    for attempt in range(3):
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw == "無" or not raw:
                return []
            names = [n.strip() for n in raw.replace("、", ",").split(",") if n.strip()]
            # filter obvious non-names (too short / too long / contains 「」etc.)
            names = [n for n in names if 1 < len(n) <= 8 and "「" not in n and "。" not in n]
            return names
        except Exception as exc:
            if attempt < 2:
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  [warn] LLM failed: {exc}", file=sys.stderr)
                return []
    return []


# ── cache helpers ─────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict[str, list[str]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, cache_path: Path):
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── edge helpers ──────────────────────────────────────────────────────────

def load_existing_edge_pairs(edge_path: Path) -> set[tuple[str, str, str]]:
    """Return set of (source, target, type) already in the file."""
    pairs: set[tuple[str, str, str]] = set()
    if not edge_path.exists():
        return pairs
    for line in edge_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pairs.add((d["source"], d["target"], d.get("type", "")))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────

def build(db_path: str, top_k: int, dry_run: bool):
    store = MemoryStore(db_path=db_path)
    db_dir = Path(db_path)
    edge_path = db_dir / GRAPH_EDGE_FILE
    cache_path = db_dir / CACHE_FILE

    # 1. load all chunks from ChromaDB
    print("Loading chunks from ChromaDB…")
    result = store.collection.get(include=["documents", "metadatas"])
    ids: list[str] = result["ids"]
    docs: list[str] = result["documents"]
    metas: list[dict] = result["metadatas"]
    total = len(ids)
    print(f"  {total} chunks found")

    # build lookup: chunk_id -> (doc, meta, order_index)
    chunk_info: dict[str, dict] = {
        cid: {"text": doc, "meta": meta, "idx": i}
        for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas))
    }

    # 2. extract characters (with resume via cache)
    cache: dict[str, list[str]] = load_cache(cache_path)
    to_process = [cid for cid in ids if cid not in cache]
    print(f"  {len(cache)} chunks already cached, {len(to_process)} to process")

    if to_process and not dry_run:
        iterator = (
            tqdm(to_process, desc="Extracting characters", unit="chunk")
            if tqdm else to_process
        )
        for i, cid in enumerate(iterator):
            text = chunk_info[cid]["text"]
            characters = extract_characters_llm(text)
            cache[cid] = characters
            # save every 50 chunks so interruption loses little work
            if (i + 1) % 50 == 0:
                save_cache(cache, cache_path)
        save_cache(cache, cache_path)
        print(f"  Cache saved → {cache_path}")

    # 3. build character → sorted chunk_ids index
    char_to_chunks: dict[str, list[str]] = {}
    for cid in ids:
        for char in cache.get(cid, []):
            char_to_chunks.setdefault(char, []).append(cid)

    # sort each character's chunk list by ingest order (chunk_index)
    for char in char_to_chunks:
        char_to_chunks[char].sort(key=lambda cid: chunk_info[cid]["idx"])

    unique_chars = len(char_to_chunks)
    print(f"\nUnique characters found: {unique_chars}")
    top10 = sorted(char_to_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for name, cids in top10:
        print(f"  {name}: {len(cids)} chunks")

    # 4. generate character edges
    existing = load_existing_edge_pairs(edge_path)
    new_edges: list[dict] = []

    for char, cids in char_to_chunks.items():
        if len(cids) < 2:
            continue
        for i, src in enumerate(cids):
            # connect to top_k nearest neighbours (by position) that share this character
            neighbours = cids[max(0, i - top_k) : i] + cids[i + 1 : i + 1 + top_k]
            src_meta = chunk_info[src]["meta"]
            for tgt in neighbours:
                key = (src, tgt, "character")
                if key in existing:
                    continue
                existing.add(key)
                new_edges.append({
                    "source": src,
                    "target": tgt,
                    "type": "character",
                    "weight": 1.0,
                    "character": char,
                    "source_title": src_meta.get("source_title", ""),
                    "chapter": src_meta.get("chapter", ""),
                })

    print(f"\nNew character edges to write: {len(new_edges)}")

    if dry_run:
        print("[dry-run] nothing written.")
        return

    # 5. append to graph_edges.jsonl
    with edge_path.open("a", encoding="utf-8") as f:
        for edge in new_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    print(f"Appended {len(new_edges)} edges → {edge_path}")


def main():
    parser = argparse.ArgumentParser(description="Build character edges for the novel graph.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--top-k", type=int, default=CHARACTER_NEIGHBORS,
                        help="Connect each chunk to K nearest neighbours per character")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and cache only, do not write edges")
    args = parser.parse_args()
    build(db_path=args.db_path, top_k=args.top_k, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
