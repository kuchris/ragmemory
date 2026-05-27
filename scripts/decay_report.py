"""
Show RagMemory decay scores without changing the database.

Examples:
    uv run python scripts/decay_report.py
    uv run python scripts/decay_report.py --sort fresh --limit 20
    uv run python scripts/decay_report.py --message-ids 560-572
    uv run python scripts/decay_report.py --output decay_report.md
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragmemory import MemoryStore
from ragmemory.memory import DECAY_SCORE_FLOOR, JOB_TYPE_STRUCTURED_EXTRACT


DEFAULT_DB_PATH = Path(os.environ.get("RAGMEMORY_DB_PATH", "./.data/chroma_db"))
DEFAULT_OUTPUT_PATH = Path(
    os.environ.get(
        "RAGMEMORY_DECAY_REPORT_PATH",
        "./.data/obsidian_memory/decay_report.md",
    )
)
TEXT_LIMIT = 120


@dataclass
class DecayReportRow:
    message_id: int
    role: str
    decay_strength: float
    retrieval_multiplier: float
    age_days: float
    access_count: int
    base_importance: float
    half_life_days: float
    pinned: bool
    last_accessed_at: str
    structured_count: int
    structured_types: str
    extraction_status: str
    review_candidate: bool
    text: str


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def parse_message_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def quiet_store(db_path: Path) -> MemoryStore:
    with contextlib.redirect_stdout(io.StringIO()):
        return MemoryStore(db_path=str(db_path))


def _age_days(store: MemoryStore, last_accessed_at: str, now: datetime) -> float:
    last_accessed = store._parse_iso_datetime(last_accessed_at)
    return max((now - last_accessed).total_seconds() / 86400, 0.0)


def structured_summary(store: MemoryStore) -> dict[int, tuple[int, str]]:
    by_message: dict[int, list[str]] = {}
    for obj in store.structured.objects.values():
        by_message.setdefault(obj.message_id, []).append(obj.type)
    return {
        message_id: (len(types), ",".join(sorted(set(types))))
        for message_id, types in by_message.items()
    }


def structured_job_statuses(store: MemoryStore) -> dict[int, str]:
    with sqlite3.connect(store.state_db) as conn:
        rows = conn.execute(
            """
            SELECT message_id, status, created_at, finished_at
            FROM jobs
            WHERE job_type = ?
            ORDER BY message_id, created_at, finished_at
            """,
            (JOB_TYPE_STRUCTURED_EXTRACT,),
        ).fetchall()
    latest: dict[int, str] = {}
    for message_id, status, _created_at, _finished_at in rows:
        latest[message_id] = status
    return latest


def extraction_status(
    structured_count: int,
    job_status: str | None,
) -> str:
    if structured_count > 0:
        return "has_structured"
    if job_status == "done":
        return "extract_done_empty"
    if job_status in {"pending", "running", "failed"}:
        return f"extract_{job_status}"
    return "unknown_or_disabled"


def load_rows(
    store: MemoryStore,
    *,
    message_ids: list[int] | None = None,
    include_tombstoned: bool = False,
    candidate_decay: float = 0.6,
) -> list[DecayReportRow]:
    where = []
    params: list[object] = []
    if not include_tombstoned:
        where.append("messages.tombstoned = 0")
        where.append("memory_metadata.tombstoned_at IS NULL")
        where.append("memory_metadata.superseded_by IS NULL")
    if message_ids:
        placeholders = ",".join("?" for _ in message_ids)
        where.append(f"messages.message_id IN ({placeholders})")
        params.extend(message_ids)
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    with sqlite3.connect(store.state_db) as conn:
        rows = conn.execute(
            f"""
            SELECT
                messages.message_id,
                messages.role,
                messages.text,
                memory_metadata.last_accessed_at,
                memory_metadata.access_count,
                memory_metadata.base_importance,
                memory_metadata.half_life_days,
                memory_metadata.pinned
            FROM messages
            JOIN memory_metadata ON memory_metadata.message_id = messages.message_id
            {where_sql}
            ORDER BY messages.message_id
            """,
            params,
        ).fetchall()

    now = datetime.now(timezone.utc)
    metadata = store._load_memory_metadata({message_id for message_id, *_ in rows})
    structured_by_message = structured_summary(store)
    jobs_by_message = structured_job_statuses(store)
    report_rows: list[DecayReportRow] = []
    for (
        message_id,
        role,
        text,
        last_accessed_at,
        access_count,
        base_importance,
        half_life_days,
        pinned,
    ) in rows:
        strength = store._decay_strength(metadata.get(message_id), now)
        multiplier = DECAY_SCORE_FLOOR + (1 - DECAY_SCORE_FLOOR) * strength
        structured_count, structured_types = structured_by_message.get(message_id, (0, ""))
        status = extraction_status(structured_count, jobs_by_message.get(message_id))
        review_candidate = (
            strength <= candidate_decay
            and status == "extract_done_empty"
        )
        report_rows.append(
            DecayReportRow(
                message_id=message_id,
                role=role,
                decay_strength=strength,
                retrieval_multiplier=multiplier,
                age_days=_age_days(store, last_accessed_at, now),
                access_count=access_count,
                base_importance=base_importance,
                half_life_days=half_life_days,
                pinned=bool(pinned),
                last_accessed_at=last_accessed_at,
                structured_count=structured_count,
                structured_types=structured_types,
                extraction_status=status,
                review_candidate=review_candidate,
                text=text,
            )
        )
    return report_rows


def sort_rows(rows: list[DecayReportRow], sort: str) -> list[DecayReportRow]:
    if sort == "fresh":
        return sorted(rows, key=lambda row: (-row.decay_strength, -row.message_id))
    if sort == "id":
        return sorted(rows, key=lambda row: row.message_id)
    return sorted(rows, key=lambda row: (row.decay_strength, row.message_id))


def print_rows(rows: list[DecayReportRow], limit: int) -> None:
    if not rows:
        print("No memory metadata rows found.")
        return
    shown = rows[: max(0, limit)]
    print(
        "message | role | decay | retrieval_x | age_d | access | "
        "base | half_life | pin | structured | extract_status | review | last_accessed | preview"
    )
    for row in shown:
        pin = "Y" if row.pinned else "N"
        review = "Y" if row.review_candidate else "N"
        structured = row.structured_types or str(row.structured_count)
        print(
            f"{row.message_id} | {row.role} | {row.decay_strength:.4f} | "
            f"{row.retrieval_multiplier:.4f} | {row.age_days:.1f} | "
            f"{row.access_count} | {row.base_importance:.2f} | "
            f"{row.half_life_days:.1f} | {pin} | {structured} | "
            f"{row.extraction_status} | {review} | {row.last_accessed_at} | {clip(row.text)}"
        )
    if len(rows) > len(shown):
        print(f"... {len(rows) - len(shown)} more row(s). Use --limit {len(rows)} to show all.")


def markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_markdown(
    rows: list[DecayReportRow],
    *,
    sort: str,
    db_path: Path,
    candidate_decay: float = 0.6,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    candidate_count = sum(1 for row in rows if row.review_candidate)
    lines = [
        "# RagMemory Decay Report",
        "",
        f"- Generated: `{now}`",
        f"- DB: `{db_path}`",
        f"- Sort: `{sort}`",
        f"- Rows: `{len(rows)}`",
        f"- Review candidates: `{candidate_count}`",
        f"- Candidate rule: `decay <= {candidate_decay}` and `extract_status == extract_done_empty`",
        "",
        "Note: `unknown_or_disabled` means no structured object exists and this report cannot prove extraction already ran.",
        "",
        "| message | role | decay | retrieval_x | age_d | access | base | half_life | pin | structured | extract_status | review | last_accessed | preview |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|---|---|:---:|---|---|",
    ]
    for row in rows:
        pin = "Y" if row.pinned else "N"
        review = "Y" if row.review_candidate else "N"
        structured = row.structured_types or str(row.structured_count)
        lines.append(
            f"| {row.message_id} | {markdown_escape(row.role)} | "
            f"{row.decay_strength:.4f} | {row.retrieval_multiplier:.4f} | "
            f"{row.age_days:.1f} | {row.access_count} | "
            f"{row.base_importance:.2f} | {row.half_life_days:.1f} | "
            f"{pin} | {markdown_escape(structured)} | {row.extraction_status} | "
            f"{review} | `{markdown_escape(row.last_accessed_at)}` | "
            f"{markdown_escape(clip(row.text))} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    rows: list[DecayReportRow],
    *,
    sort: str,
    db_path: Path,
    candidate_decay: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_markdown(
            rows,
            sort=sort,
            db_path=db_path,
            candidate_decay=candidate_decay,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show RagMemory decay scores and retrieval downweighting."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=9999)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown report path. Defaults to ./.data/obsidian_memory/decay_report.md.",
    )
    parser.add_argument("--print", action="store_true", help="Also print the table to the terminal.")
    parser.add_argument(
        "--sort",
        choices=("stale", "fresh", "id"),
        default="stale",
        help="stale shows lowest decay first.",
    )
    parser.add_argument("--message-ids", default="", help="Comma/space/range list, e.g. 560-572.")
    parser.add_argument("--include-tombstoned", action="store_true")
    parser.add_argument(
        "--candidate-decay",
        type=float,
        default=0.6,
        help="Mark review candidates at or below this decay, after extraction has completed empty.",
    )
    args = parser.parse_args(argv)

    store = quiet_store(args.db_path)
    message_ids = parse_message_ids(args.message_ids) if args.message_ids else None
    rows = load_rows(
        store,
        message_ids=message_ids,
        include_tombstoned=args.include_tombstoned,
        candidate_decay=args.candidate_decay,
    )
    ranked = sort_rows(rows, args.sort)[: max(0, args.limit)]
    write_markdown_report(
        args.output,
        ranked,
        sort=args.sort,
        db_path=args.db_path,
        candidate_decay=args.candidate_decay,
    )
    print(f"Wrote decay report: {args.output}")
    print(f"Rows: {len(ranked)}")
    print(f"Review candidates: {sum(1 for row in ranked if row.review_candidate)}")
    if args.print:
        print_rows(ranked, len(ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
