"""
Process queued RagMemory background jobs.

Run:
    uv run python scripts/run_worker.py
"""
import argparse
import time
from pathlib import Path

from ragmemory import MemoryStore
from ragmemory.memory import JOB_STATUS_DONE, JOB_STATUS_FAILED
from export_obsidian import DEFAULT_OUTPUT_PATH, export_obsidian


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RagMemory background worker."
    )
    parser.add_argument(
        "--db-path",
        default="./.data/chroma_db",
        help="RagMemory DB directory containing state.sqlite.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process available jobs once, then exit.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep when the queue is empty.",
    )
    parser.add_argument(
        "--obsidian-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Generated Obsidian mirror path to refresh after each completed job.",
    )
    parser.add_argument(
        "--config",
        default="ragmemory.local.ini",
        help="Config file for Obsidian export topic/file policies.",
    )
    parser.add_argument(
        "--retry-failed",
        type=int,
        default=3,
        metavar="N",
        help="Retry up to N previously failed compactions per sweep (0 to disable).",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    obsidian_path = Path(args.obsidian_path)
    config_path = Path(args.config)
    store = MemoryStore(db_path=str(db_path))
    reset_count = store.reset_running_jobs()
    if reset_count:
        print(f"Reset {reset_count} orphaned running job(s).")

    print("RagMemory worker started. Press Ctrl-C to stop.")
    try:
        while True:
            job = store.claim_next_job()
            if job is None:
                if args.retry_failed > 0:
                    retried = store.compact_existing_messages(
                        limit=args.retry_failed, max_attempts=3
                    )
                    if retried:
                        print(
                            f"Retried {len(retried)} failed compaction(s): "
                            f"{retried}"
                        )
                if args.once:
                    break
                time.sleep(args.sleep)
                continue

            try:
                results = store.process_background_job(job)
            except Exception as exc:
                store.complete_job(job.job_id, JOB_STATUS_FAILED, str(exc))
                print(f"Job failed: {job.job_id} {job.job_type} message={job.message_id}: {exc}")
                continue

            store.complete_job(job.job_id, JOB_STATUS_DONE)
            export_obsidian(
                db_path,
                obsidian_path,
                config_path=config_path if config_path.exists() else None,
            )
            print(
                f"Job done: {job.job_id} {job.job_type} "
                f"message={job.message_id} results={len(results)}"
            )
    except KeyboardInterrupt:
        print("RagMemory worker stopped.")


if __name__ == "__main__":
    main()
