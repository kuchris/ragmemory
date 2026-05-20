"""
View raw Chroma chunks from a memory DB.

Run:
    uv run python scripts/view_chunks.py
"""
from ragmemory.memory import MemoryStore

DB_PATH = "./chroma_structured_test"
MAX_CHARS = 700


def main():
    store = MemoryStore(db_path=DB_PATH)
    docs = store.collection.get(include=["documents", "metadatas"])

    print(f"DB: {DB_PATH}")
    print(f"Chunks: {len(docs['ids'])}")

    for index, (chunk_id, doc, meta) in enumerate(
        zip(docs["ids"], docs["documents"], docs["metadatas"]),
        start=1,
    ):
        print("\n" + "=" * 80)
        print(f"CHUNK {index}: {chunk_id}")
        print(f"METADATA: {meta}")
        print("-" * 80)
        print(doc[:MAX_CHARS])
        if len(doc) > MAX_CHARS:
            print("...")


if __name__ == "__main__":
    main()
