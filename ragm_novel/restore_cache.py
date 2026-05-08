"""Reconstruct character_cache.json from graph_edges.jsonl."""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("./chroma_novel")
edge_path = DB_PATH / "graph_edges.jsonl"
cache_path = DB_PATH / "character_cache.json"

cache: dict[str, list[str]] = {}

for line in edge_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    edge = json.loads(line)
    if edge.get("type") != "character":
        continue
    src = edge["source"]
    char = edge["character"]
    chars = cache.setdefault(src, [])
    if char not in chars:
        chars.append(char)

cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Restored {len(cache)} chunks → {cache_path}")
