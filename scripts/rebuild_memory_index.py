"""
Rebuild Chroma/BM25 chat chunks from SQLite messages.

Compact messages use messages.compact_text; all others fall back to messages.text.

Run:
    uv run python scripts/rebuild_memory_index.py
"""
import argparse
from pathlib import Path

from ragmemory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild chat retrieval index from raw/compact SQLite messages."
    )
    parser.add_argument(
        "--db-path",
        default="./.data/chroma_db",
        help="RagMemory DB directory containing state.sqlite.",
    )
    args = parser.parse_args()

    store = MemoryStore(db_path=str(Path(args.db_path)))
    chunk_count = store.rebuild_chat_memory_index()
    print(f"Rebuilt chat memory index with {chunk_count} chunk(s).")


if __name__ == "__main__":
    main()
