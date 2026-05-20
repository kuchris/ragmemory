"""
Build character-relationship graph edges — fast version.

Speedups over build_character_edges.py:
  1. Batch prompt  — N chunks per LLM call (default 8), ~8x fewer requests
  2. Async httpx   — concurrent batches with semaphore for pipelining
  3. Pre-filter    — skip chunks with no dialogue markers (「」) to save calls

Run:
    uv run python experiments/ragm_novel/build_character_edges_fast.py
    uv run python experiments/ragm_novel/build_character_edges_fast.py --dry-run
    uv run python experiments/ragm_novel/build_character_edges_fast.py --batch-size 10 --concurrency 3

Resume: safe to re-run — chunk results cached in character_cache.json.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
from pathlib import Path

import httpx

from ragmemory.memory import MemoryStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from tqdm.asyncio import tqdm as atqdm
    from tqdm import tqdm
except ImportError:
    atqdm = None
    tqdm = None

# ── LM Studio config ───────────────────────────────────────────────────────
MODEL    = os.environ.get("LMSTUDIO_MODEL",    "gemma-4-e4b-uncensored-hauhaucs-aggressive")
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY  = os.environ.get("LMSTUDIO_API_KEY",  "lm-studio")

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_DB_PATH        = "./chroma_novel"
GRAPH_EDGE_FILE        = "graph_edges.jsonl"
CACHE_FILE             = "character_cache.json"
DEFAULT_BATCH_SIZE     = 8    # chunks per LLM call
DEFAULT_CONCURRENCY    = 2    # simultaneous in-flight requests
CHARACTER_NEIGHBORS    = 5    # connect each chunk to N nearest per character
SAVE_EVERY             = 40   # save cache every N batches

# ── prompt ─────────────────────────────────────────────────────────────────
BATCH_PROMPT_TEMPLATE = """\
以下有 {n} 個小說段落，每段以 [段落N] 標記。
請分別列出每個段落中出現的人物角色名字。

輸出格式（嚴格遵守）：
1: 角色A, 角色B
2: 無
3: 角色C

規則：
- 若段落無任何角色，該行寫「無」
- 只輸出角色名，不加解釋
- 每行只輸出一個段落的結果

{chunks}"""

CHUNK_BLOCK = "[段落{i}]\n{text}\n"

# ── alias normalization ────────────────────────────────────────────────────
ALIAS_MAP: dict[str, str] = {
    # 綾小路清隆（主角）
    "清隆":           "綾小路",
    "綾小路清隆":     "綾小路",
    "綾小路君":       "綾小路",
    "綾小路同學":     "綾小路",
    "小清":           "綾小路",
    "綾小路 (我)":    "綾小路",
    "我 (綾小路)":    "綾小路",
    "清隆 (綾小路)":  "綾小路",
    "綾小路 (主角)":  "綾小路",
    "綾小路 (自稱)":  "綾小路",
    # 堀北鈴音
    "鈴音":           "堀北",
    "堀北鈴音":       "堀北",
    "堀北同學":       "堀北",
    "鈴音 (堀北)":    "堀北",
    # 龍園翔也
    "龍園翔":         "龍園",
    "龍園君":         "龍園",
    "龍園同學":       "龍園",
    "龍園 (提及)":    "龍園",
    # 一之瀨帆波
    "帆波":           "一之瀨",
    "一之瀨帆波":     "一之瀨",
    "小帆波":         "一之瀨",
    "一之瀨同學":     "一之瀨",
    "一之瀬":         "一之瀨",
    # 平田洋介
    "洋介":           "平田",
    "平田洋介":       "平田",
    "平田君":         "平田",
    # 輕井澤惠
    "惠":             "輕井澤",
    "輕井澤惠":       "輕井澤",
    "輕井澤同學":     "輕井澤",
    "輕井沢":         "輕井澤",
    # 佐倉愛裡
    "愛裡":           "佐倉",
    "佐倉愛裡":       "佐倉",
    # 篠原波瑠加
    "筱原":           "篠原",
    "波瑠加":         "篠原",
    "波琉加":         "篠原",
    "筱原皐月":       "篠原",
    # 須藤健
    "須藤健":         "須藤",
    "須藤君":         "須藤",
    # 茶柱老師
    "茶柱老師":       "茶柱",
    # 阪柳有棲（理事長）
    "阪柳有棲":       "阪柳",
    "阪柳理事長":     "阪柳",
    "阪柳同學":       "阪柳",
    # 幸村啟誠
    "啟誠":           "幸村",
    "幸村啟誠":       "幸村",
    "幸村輝彥":       "幸村",
    # 三宅明人
    "明人":           "三宅",
    "三宅明人":       "三宅",
    # 天澤一夏
    "一夏":           "天澤",
    "天澤一夏":       "天澤",
    # 真嶋老師
    "真嶋老師":       "真嶋",
    "真島老師":       "真嶋",
    "真嶋 (老師)":    "真嶋",
    # 月城
    "月城代理理事長": "月城",
    # 八神拓也
    "八神拓也":       "八神",
    # 山內春樹
    "山內春樹":       "山內",
    "春樹":           "山內",
    # 南云雅
    "南云雅":         "南云",
    # 七瀨翼
    "七瀨翼":         "七瀨",
    "七瀨同學":       "七瀨",
    # 椎名日和
    "椎名日和":       "椎名",
    # 橋本正義
    "橋本正義":       "橋本",
    "橋本君":         "橋本",
    # 山村美紀
    "山村美紀":       "山村",
    # 葛城康平
    "葛城康平":       "葛城",
    # 高圓寺六助
    "高圓寺六助":     "高圓寺",
    "高圓寺君":       "高圓寺",
    # 神崎隆二
    "神崎隆二":       "神崎",
    "神崎君":         "神崎",
    # 松下千秋
    "松下千秋":       "松下",
    # 佐藤麻耶
    "佐藤麻耶":       "佐藤",
    # 長谷部（不同於篠原）
    "長谷部波琉加":   "長谷部",
    "長谷部波瑠加":   "長谷部",
    "長谷部波琉":     "長谷部",
    # 星之宮老師
    "星之宮老師":     "星之宮",
    # 直江老師
    "直江老師":       "直江",
    # 阪上老師
    "阪上老師":       "阪上",
    # 白石飛鳥
    "白石飛鳥":       "白石",
    # 鬼頭隼
    "鬼頭隼":         "鬼頭",
}

# Names to discard — not real characters
SKIP_NAMES: set[str] = {
    "我", "(我)", "我 (敘述者)", "我 (主角)", "我 (自稱)",
    "敘述者", "(敘述者)", "敘述者 (主角)", "(敘述者/我)",
    "(我/敘述者)", "(我/堀北)", "(我/綾小路)",
    "(敘述者/主角)", "(敘述者/我)",
    "(無)", "無", "A班", "B班", "C班", "D班",
    "學生們 (群體)", "(C班學生群體)", "男人", "博士",
    "X (幕後黑手)", "知惠 (綾小路)",
}

PAREN_RE = re.compile(r"[（(][^）)]*[）)]")  # strip （主角名）(主角名) etc.


def normalize_name(raw: str) -> str:
    name = PAREN_RE.sub("", raw).strip()
    if name in SKIP_NAMES or not name:
        return ""
    return ALIAS_MAP.get(name, name)


# ── pre-filter ─────────────────────────────────────────────────────────────

def likely_has_characters(text: str) -> bool:
    """Skip chunks that almost certainly have no character dialogue/names."""
    # Light novels: dialogue always uses 「」; pure description chunks rarely have names
    return "「" in text or "」" in text or len(text) > 600


# ── async LLM call ─────────────────────────────────────────────────────────

async def call_llm_batch(
    client: httpx.AsyncClient,
    texts: list[str],
    chunk_ids: list[str],
) -> dict[str, list[str]]:
    """Send one batch request; return {chunk_id: [characters]}."""
    n = len(texts)
    chunks_block = "\n".join(
        CHUNK_BLOCK.format(i=i + 1, text=t[:800]) for i, t in enumerate(texts)
    )
    prompt = BATCH_PROMPT_TEMPLATE.format(n=n, chunks=chunks_block)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60 * n,
        "temperature": 0.0,
        "stream": False,
    }
    url = f"{BASE_URL.rstrip('/')}/chat/completions"

    for attempt in range(3):
        try:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_batch_response(raw, chunk_ids)
        except Exception as exc:
            if attempt < 2:
                await asyncio.sleep(2)
            else:
                print(f"\n  [warn] batch failed: {exc}", file=sys.stderr)
                return {cid: [] for cid in chunk_ids}
    return {cid: [] for cid in chunk_ids}


def _parse_batch_response(raw: str, chunk_ids: list[str]) -> dict[str, list[str]]:
    result = {cid: [] for cid in chunk_ids}
    pattern = re.compile(r"^(\d+)\s*[:：]\s*(.+)$")
    for line in raw.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(chunk_ids):
            continue
        value = m.group(2).strip()
        if value == "無" or not value:
            continue
        raw_names = [n.strip() for n in value.replace("、", ",").split(",") if n.strip()]
        names: list[str] = []
        seen: set[str] = set()
        for rn in raw_names:
            name = normalize_name(rn)
            if not name or name in seen:
                continue
            if not (1 < len(name) <= 8) or "「" in name or "。" in name:
                continue
            seen.add(name)
            names.append(name)
        result[chunk_ids[idx]] = names
    return result


# ── cache helpers ──────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── edge helpers ───────────────────────────────────────────────────────────

def load_existing_edge_pairs(edge_path: Path) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    if not edge_path.exists():
        return pairs
    for line in edge_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pairs.add((d["source"], d["target"], d.get("type", "")))
    return pairs


# ── async extraction loop ──────────────────────────────────────────────────

async def extract_all(
    to_process: list[tuple[str, str]],  # [(chunk_id, text), ...]
    cache: dict[str, list[str]],
    cache_path: Path,
    batch_size: int,
    concurrency: int,
):
    semaphore = asyncio.Semaphore(concurrency)
    batches = [
        to_process[i : i + batch_size]
        for i in range(0, len(to_process), batch_size)
    ]
    completed = 0

    bar = tqdm(total=len(batches), desc="LLM batches", unit="batch") if tqdm else None

    async def process_batch(batch):
        nonlocal completed
        ids   = [cid for cid, _ in batch]
        texts = [txt for _, txt in batch]
        async with semaphore:
            result = await call_llm_batch(client, texts, ids)
        cache.update(result)
        completed += 1
        if completed % SAVE_EVERY == 0:
            save_cache(cache, cache_path)
        if bar:
            bar.update(1)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[process_batch(b) for b in batches])

    if bar:
        bar.close()
    save_cache(cache, cache_path)


# ── main ───────────────────────────────────────────────────────────────────

def build(db_path: str, top_k: int, batch_size: int, concurrency: int, dry_run: bool):
    store    = MemoryStore(db_path=db_path)
    db_dir   = Path(db_path)
    edge_path  = db_dir / GRAPH_EDGE_FILE
    cache_path = db_dir / CACHE_FILE

    print("Loading chunks from ChromaDB…")
    result = store.collection.get(include=["documents", "metadatas"])
    ids:   list[str]  = result["ids"]
    docs:  list[str]  = result["documents"]
    metas: list[dict] = result["metadatas"]
    print(f"  {len(ids)} chunks total")

    chunk_info = {
        cid: {"text": doc, "meta": meta, "idx": i}
        for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas))
    }

    # pre-filter
    filtered_ids = [cid for cid in ids if likely_has_characters(chunk_info[cid]["text"])]
    skipped = len(ids) - len(filtered_ids)
    print(f"  Pre-filter: {skipped} chunks skipped (no dialogue), {len(filtered_ids)} to check")

    # resume from cache
    cache = load_cache(cache_path)
    to_process = [
        (cid, chunk_info[cid]["text"])
        for cid in filtered_ids
        if cid not in cache
    ]
    print(f"  Cache: {len(cache)} done, {len(to_process)} remaining")

    estimated_calls = (len(to_process) + batch_size - 1) // batch_size
    print(f"  Batches to send: {estimated_calls} (batch_size={batch_size}, concurrency={concurrency})")

    if to_process and not dry_run:
        t0 = time.time()
        asyncio.run(extract_all(to_process, cache, cache_path, batch_size, concurrency))
        elapsed = time.time() - t0
        print(f"  Extraction done in {elapsed:.0f}s ({elapsed/max(len(to_process),1):.2f}s/chunk)")
        print(f"  Cache saved → {cache_path}")

    # also mark skipped/filtered chunks as empty so they are not re-checked
    for cid in ids:
        if cid not in cache:
            cache[cid] = []

    # build character → sorted chunk list
    char_to_chunks: dict[str, list[str]] = {}
    for cid in ids:
        for char in cache.get(cid, []):
            char_to_chunks.setdefault(char, []).append(cid)

    for char in char_to_chunks:
        char_to_chunks[char].sort(key=lambda cid: chunk_info[cid]["idx"])

    print(f"\nUnique characters: {len(char_to_chunks)}")
    for name, cids in sorted(char_to_chunks.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  {name}: {len(cids)} chunks")

    # generate edges
    existing  = load_existing_edge_pairs(edge_path)
    new_edges: list[dict] = []

    for char, cids in char_to_chunks.items():
        if len(cids) < 2:
            continue
        for i, src in enumerate(cids):
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

    print(f"\nNew character edges: {len(new_edges)}")

    if dry_run:
        print("[dry-run] nothing written.")
        return

    with edge_path.open("a", encoding="utf-8") as f:
        for edge in new_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    print(f"Appended → {edge_path}")


def main():
    parser = argparse.ArgumentParser(description="Build character edges (fast version).")
    parser.add_argument("--db-path",     default=DEFAULT_DB_PATH)
    parser.add_argument("--top-k",       type=int, default=CHARACTER_NEIGHBORS)
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE,
                        help="Chunks per LLM call (default 8)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Concurrent in-flight requests (default 2)")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()
    build(
        db_path=args.db_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
