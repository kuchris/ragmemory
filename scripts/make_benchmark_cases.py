"""
Create a local retrieval benchmark corpus from existing structured memory.

The output is written under .data by default because it may contain private
project names, paths, and workflow details from your real memory store.

Run:
    uv run python scripts/make_benchmark_cases.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / ".data" / "chroma_db"
DEFAULT_OUTPUT = ROOT / ".data" / "bench_retrieval" / "real_cases.json"
SECRET_RE = re.compile(r"(api[_ -]?key|secret|token|sk-[a-z0-9_-]{8,}|nvapi-[a-z0-9_-]{8,})", re.I)
TYPE_QUOTAS = {
    "decision": 10,
    "preference": 5,
    "constraint": 5,
    "config": 4,
    "code_reference": 3,
    "open_question": 3,
}


def _load_objects(db_path: Path) -> list[dict]:
    path = db_path / "structured_memory.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    objects = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            objects.append(json.loads(line))
    return objects


def _safe_text(value: str) -> bool:
    return bool(value.strip()) and SECRET_RE.search(value) is None


def _dedupe_key(obj: dict) -> tuple[str, str]:
    return obj.get("type", ""), re.sub(r"\s+", " ", obj.get("summary", "")).strip().lower()


def _subject(obj: dict) -> str:
    tags = [tag for tag in obj.get("tags", []) if tag]
    if tags:
        return " / ".join(tags[:3])
    summary = re.sub(r"\s+", " ", obj.get("summary", "")).strip()
    return " ".join(summary.split()[:5])


def _query(obj: dict) -> str:
    subject = _subject(obj)
    obj_type = obj.get("type")
    if obj_type == "decision":
        return f"what did we decide about {subject}?"
    if obj_type == "preference":
        return f"what preference was recorded about {subject}?"
    if obj_type == "constraint":
        return f"what constraint applies to {subject}?"
    if obj_type == "config":
        return f"what config detail should I remember for {subject}?"
    if obj_type == "code_reference":
        return f"what code reference was saved for {subject}?"
    if obj_type == "open_question":
        return f"what open question remains about {subject}?"
    return f"what should I remember about {subject}?"


def _must_contain(obj: dict) -> str:
    summary = re.sub(r"\s+", " ", obj.get("summary", "")).strip()
    words = summary.split()
    if len(words) <= 10:
        return summary
    return " ".join(words[:10])


def build_cases(objects: list[dict], limit: int) -> list[dict]:
    by_type: dict[str, list[dict]] = {name: [] for name in TYPE_QUOTAS}
    seen = set()
    for obj in sorted(
        objects,
        key=lambda item: (item.get("importance", 0), item.get("message_id", 0)),
        reverse=True,
    ):
        obj_type = obj.get("type")
        if obj_type not in by_type:
            continue
        summary = obj.get("summary", "")
        if not _safe_text(summary):
            continue
        key = _dedupe_key(obj)
        if key in seen:
            continue
        seen.add(key)
        by_type[obj_type].append(obj)

    selected = []
    for obj_type, quota in TYPE_QUOTAS.items():
        selected.extend(by_type[obj_type][:quota])

    if len(selected) < limit:
        already = {obj.get("id") for obj in selected}
        extras = [
            obj
            for items in by_type.values()
            for obj in items
            if obj.get("id") not in already
        ]
        selected.extend(extras[: limit - len(selected)])

    cases = []
    for index, obj in enumerate(selected[:limit], start=1):
        cases.append(
            {
                "name": f"real_{index:02d}_{obj.get('type')}_{obj.get('message_id')}",
                "query": _query(obj),
                "top_k": 10,
                "must_contain": [_must_contain(obj)],
                "source_message_id": obj.get("message_id"),
                "source_type": obj.get("type"),
                "source_tags": obj.get("tags", []),
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local RagMemory benchmark cases from structured memory."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    objects = _load_objects(Path(args.db_path))
    cases = build_cases(objects, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} case(s) to {output}")


if __name__ == "__main__":
    main()
