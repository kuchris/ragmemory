"""
Verify BM25 additions rebuild lazily on search.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_bm25_lazy_rebuild.py
"""
from ragmemory.memory import BM25Index

index = BM25Index()

index.add("a", "alpha memory")
index.add("b", "beta memory")
index.add("c", "gamma memory")
assert index._dirty is True
assert index._bm25 is None
assert index.search("alpha", top_k=1) == ["a"]
assert index._dirty is False
assert index._bm25 is not None

previous_bm25 = index._bm25
index.add("d", "delta memory")
assert index._dirty is True
assert index._bm25 is previous_bm25
assert index.search("delta", top_k=1) == ["d"]
assert index._dirty is False
assert index.search("delta", top_k=0) == []

index.build(["a", "b", "c"], ["alpha memory", "beta memory", "gamma memory"])
assert index._dirty is False
assert index.search("gamma", top_k=1) == ["c"]

print("BM25 lazy rebuild test passed.")
