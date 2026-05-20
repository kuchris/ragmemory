"""
Ask retrieval questions against an existing memory DB.

Run:
    uv run python scripts/ask_memory.py
"""
from ragmemory.memory import MemoryStore

DB_PATH = "./chroma_structured_test"

QUESTIONS = [
    "what was the opencode config?",
    "why did we keep memory.py separate?",
    "what should not be merged?",
    "what was my preference about ragm_mcp?",
    "where are raw chunks and structured memory objects saved?",
]


def main():
    store = MemoryStore(db_path=DB_PATH)

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)
        print(store.build_context(question))


if __name__ == "__main__":
    main()
