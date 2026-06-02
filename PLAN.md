# Plan: Hybrid RagMemory and Codex-Style Memory

## Goal

Build a hybrid memory workflow that keeps RagMemory's automatic capture,
retrieval, decay, Obsidian export, and audit storage, while adding Codex-style
task-group summaries for cheaper and more reliable agent reasoning.

The first target is not to replace RagMemory retrieval. The first target is to
generate a small, structured "agent memory index" that tells an agent what to
reuse, where the evidence lives, and what mistakes to avoid.

## Assumptions

- RagMemory remains the source of truth for raw messages, structured memory,
  events, tombstones, decay metadata, and Obsidian export.
- Codex-style memory is a derived layer, not a second source of truth.
- Generated Codex-style summaries must point back to RagMemory message IDs,
  events, structured objects, or exported notes.
- The system should prefer small routed summaries before injecting many raw
  chunks into the prompt.
- Manual edits should be possible, but manual notes should be clearly marked
  and should never become executable instructions.

## Desired Output Shape

Add a derived folder like:

```text
.data/codex_memory/
  memory_summary.md
  MEMORY.md
  INDEX.json
  task_groups/
    ct_mv_retry.md
    ragmemory_obsidian.md
  evidence/
    message-refs.jsonl
```

The key file is `MEMORY.md`, using the Codex-style shape:

```md
# Task Group: <name>
scope: <when to use this memory>
applies_to: <repo/path/topic boundaries and reuse rule>

## Task 1: <task>

### evidence

- message_id: ...
- event_id: ...
- obsidian_note: ...

### keywords

- exact symbols, paths, terms, error text

## User preferences

- when the user said ... -> answer/do ...

## Reusable knowledge

- verified conclusion with source pointer

## Failures and how to do differently

- symptom -> cause -> fix
```

## Phase 1: Read-Only Export Prototype

Success criteria:

- A script can generate `.data/codex_memory/MEMORY.md` from existing RagMemory
  data without changing normal recall behavior.
- Each generated task group includes `scope`, `applies_to`, `keywords`,
  `User preferences`, `Reusable knowledge`, and `Failures and how to do
  differently`.
- Every reusable claim includes at least one source pointer back to RagMemory
  storage.
- The export is deterministic enough that repeated runs do not create noisy
  diffs.

Implementation sketch:

- Add `scripts/export_codex_memory.py`.
- Read from `state.sqlite`, `structured_memory.jsonl`, and existing metadata.
- Group memories by repo/cwd, topic, and repeated task family.
- Use existing compact text when `compact_status = ok`; otherwise fall back to
  raw text.
- Write Markdown only under `.data/codex_memory/`.

Verification:

```powershell
uv run python scripts/export_codex_memory.py --db-path ./.data/chroma_db --output ./.data/codex_memory
Get-ChildItem .\.data\codex_memory
```

## Phase 2: Add a Tiny Routing Index

Success criteria:

- `INDEX.json` maps query terms, repo paths, and topics to task-group files.
- A lookup can find likely task groups without reading the whole `MEMORY.md`.
- The index includes estimated read cost and staleness hints.

Example:

```json
{
  "ct_mv_retry": {
    "file": "task_groups/ct_mv_retry.md",
    "project": "C:\\Users\\alten\\Desktop\\ku\\ct",
    "topics": ["MV retry", "MarsHandler", "W:", "backup", "Ths15", "Rs2"],
    "read_cost": "low",
    "stability": {
      "stable": ["MarsHandler owns MV retry runtime behavior"],
      "recheck": ["current W: config values"]
    }
  }
}
```

Verification:

```powershell
uv run python scripts/export_codex_memory.py --db-path ./.data/chroma_db --output ./.data/codex_memory --index
Get-Content .\.data\codex_memory\INDEX.json
```

## Phase 3: Task-Card Recall Mode

Success criteria:

- Before normal chunk injection, RagMemory can retrieve one to five compact
  Codex-style task cards.
- Task-card recall uses small token budgets and does not duplicate normal MCP
  recall/save behavior.
- The agent sees task boundaries and failure lessons before raw supporting
  chunks.

Implementation sketch:

- Add a recall mode such as `codex_task_cards`.
- Query `INDEX.json` and generated task-group files using cwd, prompt terms,
  exact symbols, and recent conversation metadata.
- Inject only the matching task-card summary first.
- Keep existing vector/BM25 recall as fallback evidence retrieval.

Verification:

```powershell
uv run python scripts/preview_codex_recall.py --query "MarsHandler MV retry W:" --cwd C:\Users\alten\Desktop\ku\ct
```

## Phase 4: Evidence Drilldown

Success criteria:

- A task card can point to exact RagMemory messages, events, and Obsidian notes.
- The agent can open evidence only when needed instead of reading everything.
- Generated citations or debug output show which memory items were used.

Implementation sketch:

- Add `evidence/message-refs.jsonl`.
- Include `message_id`, `created_at`, `cwd`, `source`, `compact_status`,
  `structured_memory_id`, and optional Obsidian note path.
- Add a small CLI to resolve a task-group evidence pointer into raw/compact
  source text.

Verification:

```powershell
uv run python scripts/resolve_codex_memory_evidence.py --id ct_mv_retry --limit 5
```

## Phase 5: Manual Review and Correction Layer

Success criteria:

- Users can add small correction notes without editing generated files.
- Corrections are marked as manual/ad-hoc in generated summaries.
- Manual notes are treated as information, not instructions to execute.

Implementation sketch:

```text
.data/codex_memory/extensions/ad_hoc/notes/
  2026-06-02-ct-mv-retry-correction.md
```

The export script should merge these notes into generated summaries with a
marker such as `[ad-hoc note]`.

Verification:

```powershell
uv run python scripts/export_codex_memory.py --db-path ./.data/chroma_db --output ./.data/codex_memory
rg "ad-hoc note" .\.data\codex_memory
```

## Phase 6: Quality Checks

Success criteria:

- Generated task groups do not mix unrelated repos or runtime surfaces.
- Claims about current code are marked as `recheck` unless recently verified.
- Failure lessons are preserved when present.
- The generated output stays small enough to be useful as prompt context.

Suggested checks:

```powershell
uv run python scripts/check_codex_memory.py --output ./.data/codex_memory
uv run python tests/test_codex_memory_export.py
```

Checks to include:

- no task group without evidence pointers
- no task group crossing unrelated cwd values unless explicitly marked
- no generated file over a configured size limit
- no duplicate task IDs
- no raw secret-looking values copied into generated Markdown

## Open Design Questions

- Should the generated Codex memory live only under `.data/`, or should a
  public-safe sample live in `docs/`?
- Should topic grouping reuse `topic_taxonomy.json`, or should Codex task
  groups have a separate taxonomy?
- Should task-card recall run automatically in hooks, or only through a preview
  command until the format is stable?
- How much should LLM extraction decide task grouping versus deterministic
  grouping by `cwd`, topic, and repeated symbols?

## Minimal First Experiment

Start with one read-only script:

```powershell
uv run python scripts/export_codex_memory.py --db-path ./.data/chroma_db --output ./.data/codex_memory --limit 20
```

Then manually inspect:

```powershell
Get-Content .\.data\codex_memory\MEMORY.md
```

If the generated task groups are useful, add `INDEX.json`. Only after that,
try hook-based task-card recall.
