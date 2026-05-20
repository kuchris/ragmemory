import argparse
import json
from pathlib import Path


def load_events(path: Path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def matches(event: dict, args) -> bool:
    if args.event and event.get("event") != args.event:
        return False
    if args.message_id is not None:
        values = event.get("message_ids", [])
        if not isinstance(values, list):
            values = []
        if event.get("message_id") != args.message_id and args.message_id not in values:
            return False
    if args.since and event.get("ts", "") < args.since:
        return False
    if args.until and event.get("ts", "") > args.until:
        return False
    return True


def summarize(event: dict) -> str:
    parts = [
        event.get("ts", ""),
        event.get("event", ""),
    ]
    if "message_id" in event:
        parts.append(f"message_id={event['message_id']}")
    if "message_ids" in event:
        parts.append(f"message_ids={event['message_ids']}")
    if "result_count" in event:
        parts.append(f"result_count={event['result_count']}")
    if "kept_count" in event:
        parts.append(f"kept={event['kept_count']}")
    if "dropped_count" in event:
        parts.append(f"dropped={event['dropped_count']}")
    if "job_id" in event:
        parts.append(f"job_id={event['job_id']}")
    return " | ".join(str(part) for part in parts if part != "")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect RagMemory events.jsonl.")
    parser.add_argument(
        "--db-path",
        default="./.data/chroma_structured_test",
        help="RagMemory DB directory containing events.jsonl.",
    )
    parser.add_argument("--event", help="Only show this event type.")
    parser.add_argument("--message-id", type=int, help="Only show events for this message id.")
    parser.add_argument("--since", help="Only show events at or after this ISO timestamp.")
    parser.add_argument("--until", help="Only show events at or before this ISO timestamp.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum events to print.")
    parser.add_argument("--json", action="store_true", help="Print full JSON records.")
    args = parser.parse_args(argv)

    events_path = Path(args.db_path) / "events.jsonl"
    events = [event for event in load_events(events_path) if matches(event, args)]
    if args.limit >= 0:
        events = events[-args.limit:]

    for event in events:
        if args.json:
            print(json.dumps(event, ensure_ascii=False))
        else:
            print(summarize(event))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
