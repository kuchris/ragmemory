"""
Check the live Obsidian graph export stays semantic and small.

This script does not write to the RagMemory DB. By default it refreshes the
Obsidian mirror from the DB, then checks the generated Markdown graph shape.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_obsidian import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TOPIC_DENYLIST,
    export_obsidian,
)


DEFAULT_DB_PATH = Path("./.data/chroma_db")
DEFAULT_CONFIG_PATH = Path("./ragmemory.local.ini")


def note_stems(root: Path) -> set[str]:
    stems = set()
    for note in root.rglob("*.md"):
        stems.add(note.relative_to(root).with_suffix("").as_posix())
        stems.add(note.stem)
    return stems


def missing_wikilinks(root: Path) -> list[tuple[str, str]]:
    stems = note_stems(root)
    missing = []
    for note in root.rglob("*.md"):
        text = note.read_text(encoding="utf-8")
        body = text.split("---", 2)[2] if text.startswith("---") and text.count("---") >= 2 else text
        for target in re.findall(r"\[\[([^|\]#]+)", body):
            if target not in stems:
                missing.append((note.relative_to(root).as_posix(), target))
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check generated Obsidian graph quality.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-topic-hubs", type=int, default=300)
    parser.add_argument("--no-export", action="store_true", help="Check existing Markdown without refreshing it.")
    args = parser.parse_args(argv)

    if not args.no_export:
        export_obsidian(args.db_path, args.output, config_path=args.config)

    topics_dir = args.output / "topics"
    topic_hubs = sorted(path.stem for path in topics_dir.glob("*.md")) if topics_dir.exists() else []
    errors = []

    if len(topic_hubs) > args.max_topic_hubs:
        errors.append(f"topic hub count {len(topic_hubs)} exceeds ceiling {args.max_topic_hubs}")

    denied_hubs = sorted(tag for tag in DEFAULT_TOPIC_DENYLIST if (topics_dir / f"{tag}.md").exists())
    if denied_hubs:
        errors.append("denylisted topic hubs exist: " + ", ".join(denied_hubs))

    missing = missing_wikilinks(args.output)
    if missing:
        sample = "; ".join(f"{source} -> {target}" for source, target in missing[:5])
        errors.append(f"missing wikilink targets: {sample}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print(
        f"OK: {len(topic_hubs)} topic hub(s), no denylisted hubs, no phantom wikilinks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
