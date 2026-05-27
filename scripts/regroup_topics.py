"""
Queue or run an LLM topic-regrouping job for the Obsidian mirror.

Run:
    uv run python scripts/regroup_topics.py --queue --db-path ./.data/chroma_db
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ragmemory import MemoryStore
from ragmemory.topics import (
    TopicRegroupOptions,
    build_topic_llm_options,
    topic_taxonomy_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue or run RagMemory LLM topic regrouping."
    )
    parser.add_argument(
        "--db-path",
        default="./.data/chroma_db",
        help="RagMemory DB directory containing state.sqlite.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--queue", action="store_true", help="Queue a worker topic_regroup job.")
    mode.add_argument("--run", action="store_true", help="Run topic regrouping immediately.")
    args = parser.parse_args()

    store = MemoryStore(db_path=str(Path(args.db_path)))
    options = TopicRegroupOptions.from_env()
    if not options.enabled:
        print("Topic regroup disabled: set [topic_regroup] enable = true.")
        return
    if args.queue:
        job_id = store.enqueue_topic_regroup()
        if job_id:
            print(f"Queued topic regroup job: {job_id}")
        else:
            print("Topic regroup job already pending or running.")
        return

    try:
        path = store.regroup_topics()
    except Exception as exc:
        llm_options = build_topic_llm_options(options)
        existing = topic_taxonomy_path(Path(args.db_path))
        print(f"Topic regroup failed: {exc}")
        print(
            "Config: "
            f"provider={llm_options.provider} "
            f"model={llm_options.model} "
            f"max_tokens={options.max_tokens} "
            f"extra_body={llm_options.extra_body}"
        )
        if existing.exists():
            print(f"Kept existing taxonomy: {existing}")
        else:
            print(f"No taxonomy written: {existing}")
        raise SystemExit(1) from exc
    print(f"Wrote topic taxonomy: {path}")


if __name__ == "__main__":
    main()
