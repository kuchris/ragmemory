"""
Run the fixed golden retrieval corpus.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_golden_retrieval.py
"""
import json
import shutil
from pathlib import Path

import ragmemory.memory as memory
from ragmemory.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "tests" / "golden" / "retrieval_cases.json"
DB_PATH = ROOT / ".data" / "chroma_golden_retrieval_test"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
original_recent = memory.RECENT_MESSAGES
memory.RECENT_MESSAGES = 0

try:
    store = MemoryStore(db_path=str(DB_PATH))

    for message in corpus["messages"]:
        store.add_message(message["role"], message["text"], extract_structured=False)

    for case in corpus["cases"]:
        bundle = store.build_context_bundle(case["query"])
        retrieved_text = "\n".join(
            chunk.text for chunk in bundle.retrieved + bundle.ledger_recovered
        )

        missing = [
            expected
            for expected in case["must_contain"]
            if expected not in retrieved_text
        ]
        if missing:
            raise AssertionError(
                f"{case['name']} failed. Missing {missing!r} from retrieved text:\n"
                f"{retrieved_text}"
            )
        assert bundle.query == case["query"]
        assert bundle.token_budget > 0
        assert bundle.tokens_used <= bundle.token_budget

        print(f"PASS {case['name']}")
finally:
    memory.RECENT_MESSAGES = original_recent

print("Golden retrieval test passed.")
