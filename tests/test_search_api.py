"""
Verify the public search API returns typed result objects.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_search_api.py
"""
import shutil
from pathlib import Path

from ragmemory import MemoryStore, SearchResult

DB_PATH = Path("./.data/chroma_search_api_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
store.add_message(
    "user",
    "Search API contract: return typed SearchResult objects for raw chunks.",
    extract_structured=False,
)
store.add_message(
    "assistant",
    "The result source should be raw_chunk and metadata should include importance.",
    extract_structured=False,
)

results = store.search("what should search return?", top_k=2)

assert results
assert all(isinstance(result, SearchResult) for result in results)
assert len(results) <= 2

top = results[0]
assert top.item_id
assert top.item_type == "chunk"
assert top.text
assert top.score > 0
assert top.message_id is not None
assert top.source == "raw_chunk"
assert top.metadata == {"importance": 0.5}

assert store.search("what should search return?", top_k=0) == []

print("Search API test passed.")
