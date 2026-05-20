"""
Run the fixed golden retrieval corpus.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_golden_retrieval.py
"""
import json
import shutil
from pathlib import Path

from ragmemory.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "tests" / "golden" / "retrieval_cases.json"
DB_PATH = ROOT / ".data" / "chroma_golden_retrieval_test"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
store = MemoryStore(db_path=str(DB_PATH))

for message in corpus["messages"]:
    store.add_message(message["role"], message["text"], extract_structured=False)

for case in corpus["cases"]:
    results = store.retrieve(case["query"], top_k=case.get("top_k", 3))
    retrieved_text = "\n".join(result.text for result in results)

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

    print(f"PASS {case['name']}")

print("Golden retrieval test passed.")
