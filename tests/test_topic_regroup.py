"""
Verify topic taxonomy validation and worker job dispatch.

Run:
    uv run python tests/test_topic_regroup.py
"""
import json
import os
import shutil
from pathlib import Path

from ragmemory.memory import BackgroundJob, JOB_TYPE_TOPIC_REGROUP, MemoryStore
from ragmemory.topics import (
    _extract_json_object,
    build_topic_llm_options,
    save_validated_topic_taxonomy,
    TopicRegroupOptions,
)


DB_PATH = Path("./.data/topic_regroup_test")
TAXONOMY_PATH = DB_PATH / "topic_taxonomy.json"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)
DB_PATH.mkdir(parents=True)

previous = {
    "generated_at": "old",
    "model": "old-model",
    "source_object_count": 1,
    "topics": [
        {
            "id": "old-topic",
            "title": "Old topic",
            "description": "Keep this if validation fails.",
            "aliases": [],
            "structured_ids": ["sm_valid"],
        }
    ],
}
save_validated_topic_taxonomy(TAXONOMY_PATH, previous, {"sm_valid"})
before = TAXONOMY_PATH.read_text(encoding="utf-8")

try:
    save_validated_topic_taxonomy(
        TAXONOMY_PATH,
        {
            "topics": [
                {
                    "id": "bad-topic",
                    "title": "Bad topic",
                    "description": "",
                    "aliases": [],
                    "structured_ids": ["sm_missing"],
                }
            ]
        },
        {"sm_valid"},
    )
except ValueError:
    pass
else:
    raise AssertionError("invalid taxonomy should fail validation")

assert TAXONOMY_PATH.read_text(encoding="utf-8") == before
assert _extract_json_object("prefix {\"topics\": []} suffix") == "{\"topics\": []}"
try:
    _extract_json_object("")
except ValueError as exc:
    assert "empty response" in str(exc)
else:
    raise AssertionError("empty topic response should fail")

env_keys = [
    "RAGMEMORY_TOPIC_PROVIDER",
    "RAGMEMORY_TOPIC_MODEL",
    "RAGMEMORY_TOPIC_THINKING",
    "RAGMEMORY_LLM_OPENCODE_GO_THINKING",
]
saved_env = {key: os.environ.get(key) for key in env_keys}
try:
    os.environ["RAGMEMORY_TOPIC_PROVIDER"] = "opencode_go"
    os.environ["RAGMEMORY_TOPIC_MODEL"] = "deepseek-v4-flash"
    os.environ["RAGMEMORY_TOPIC_THINKING"] = "disabled"
    os.environ["RAGMEMORY_LLM_OPENCODE_GO_THINKING"] = "enabled"
    topic_options = TopicRegroupOptions.from_env()
    llm_options = build_topic_llm_options(topic_options)
    assert llm_options.extra_body == {"thinking": {"type": "disabled"}}
finally:
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

store = MemoryStore(db_path=str(DB_PATH))
job_id = store.enqueue_topic_regroup()
assert job_id
deduped = store.enqueue_topic_regroup()
assert deduped is None

expected_path = str(TAXONOMY_PATH)
store.regroup_topics = lambda: expected_path
result = store.process_background_job(
    BackgroundJob(
        job_id="job_topic",
        job_type=JOB_TYPE_TOPIC_REGROUP,
        message_id=0,
        attempts=1,
    )
)
assert result == [expected_path]

saved = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
assert saved["topics"][0]["id"] == "old-topic"

print("Topic regroup test passed.")
