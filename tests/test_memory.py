"""
Automated test: feeds long messages into memory, then checks retrieval.
Run with: uv run python tests/test_memory.py
"""
import shutil
from pathlib import Path
from ragmemory.memory import MemoryStore

DB_PATH = Path("./chroma_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))

# Feed the full plan as one big user message
plan = Path("experiments/llamac_plan.md").read_text(encoding="utf-8")
print("=== Feeding full plan ===\n")
store.add_message("user", plan)
print(f"Plan length: {len(plan)} chars\n")

MESSAGES = [
    ("user", "My name is David and I live in Hong Kong. I work as a machine learning engineer at a startup focused on LLM infrastructure."),
    ("assistant", "Nice to meet you David! Working on LLM infrastructure in Hong Kong sounds exciting. What kind of projects are you working on?"),
    ("user", """We are building a RAG memory system for long chat sessions. The main challenge is that as conversations grow,
the context window fills up and the model loses track of earlier parts of the conversation.
Our approach uses tiered memory: a hot layer with recent raw messages, a warm layer with structured facts,
and a cold archive layer for older content that gets compressed out.
Retrieval is done via hybrid search combining dense embeddings and BM25 keyword matching."""),
    ("assistant", "That's a sophisticated architecture. The tiered approach makes a lot of sense for managing context efficiently."),
    ("user", """The compression step uses a draft model to do extractive compression — it never rewrites, only selects what to keep.
Everything removed goes into a removal ledger so it can be recovered if needed.
We call this recoverable compression. The key insight is that you're not deleting memory, you're paging it in and out.
Performance target is under 100ms total overhead per turn, including chunking, embedding, and retrieval."""),
    ("assistant", "Recoverable compression is a smart design. The removal ledger prevents permanent information loss."),
    ("user", "We are using nomic-embed-text for embeddings and ChromaDB as the vector store. The main model is still TBD but likely a llama-based model via llama.cpp."),
    ("assistant", "Good choices for local deployment. nomic-embed-text has strong retrieval performance for its size."),
    ("user", "One concern is VRAM. Since the KV cache grows with context length, keeping the prompt short via RAG compression should significantly reduce VRAM usage."),
    ("assistant", "Exactly — a 32K context vs 4K context can mean 4-8x difference in KV cache size. Big win for local inference."),
]

print("=== Feeding short messages ===\n")
for role, text in MESSAGES:
    store.add_message(role, text)
    print(f"{role.upper()}: {text[:60]}...")

print("\n=== Retrieval Tests ===\n")

queries = [
    "where does David live?",
    "what is the compression approach?",
    "what embedding model are they using?",
    "VRAM and context window",
    "removal ledger",
]

for q in queries:
    print(f"QUERY: {q}")
    results = store.retrieve(q, top_k=2)
    for i, r in enumerate(results):
        print(f"  [{i+1}] importance={r.importance} | {r.text[:100]}...")
    print()

print("=== Compression Test ===\n")
from ragmemory.memory import CONTEXT_TOKEN_BUDGET
print(f"Token budget: {CONTEXT_TOKEN_BUDGET}")
ctx = store.build_context("what did we decide about compression?")
print(f"Ledger size after compression: {len(store.ledger)}")
print(f"\nFinal context ({store._estimate_tokens(ctx)} tokens):\n{ctx[:600]}...")
