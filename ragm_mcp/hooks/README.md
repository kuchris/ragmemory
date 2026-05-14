# RagMemory Codex Hooks

These hooks make Codex use RagMemory automatically.

They are installed through the user-level Codex config:

```text
%USERPROFILE%\.codex\config.toml
```

Do not commit that file. It is machine-local and contains your local checkout path.

## What The Hooks Do

`UserPromptSubmit`

- Runs when the user submits a prompt.
- Saves the user prompt to RagMemory.
- Retrieves relevant memory.
- Returns it to Codex as `additionalContext`.

`Stop`

- Runs when Codex is about to end its turn.
- Saves the latest assistant message to RagMemory.
- Writes a small debug record to `ragm_mcp/chroma_db/hook_debug.jsonl`.

Both hooks write through the same backend as the MCP server:

```text
ragm_mcp/chroma_db/state.json
ragm_mcp/chroma_db/chroma.sqlite3
```

The MCP server does not need to be running for these hook scripts. They import the local `MemoryStore` directly.

## Files

```text
ragm_mcp/hooks/user_prompt_submit.py
ragm_mcp/hooks/stop.py
```

## Install For Codex On Windows

Add this to:

```text
%USERPROFILE%\.codex\config.toml
```

Replace `C:\path\to\ragmemory` with your local repo path.

```toml
[features]
codex_hooks = true

[[hooks.UserPromptSubmit]]

[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = 'cmd /d /c "set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& cd /d C:\path\to\ragmemory&& uv run python ragm_mcp\hooks\user_prompt_submit.py"'
timeout = 30
statusMessage = "rag-memory: recalling relevant memory"

[[hooks.Stop]]

[[hooks.Stop.hooks]]
type = "command"
command = 'cmd /d /c "set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& cd /d C:\path\to\ragmemory&& uv run python ragm_mcp\hooks\stop.py"'
timeout = 30
statusMessage = "rag-memory: saving assistant message"
```

Restart Codex after editing `config.toml`.

Codex may add entries under `[hooks.state]`. If a hook is present there but not running, check that it has:

```toml
enabled = true
```

If Codex changes the hook command, it may disable the hook again until it is trusted.

## Verify

Ask Codex any question. At the start of the next turn, you should see injected context like:

```text
=== RagMemory Context ===
...
=== End RagMemory Context ===
```

After Codex answers, check:

```text
ragm_mcp/chroma_db/state.json
ragm_mcp/chroma_db/hook_debug.jsonl
```

Expected result:

- `state.json` gets a new `user` entry from `UserPromptSubmit`.
- `state.json` gets a new `assistant` entry from `Stop`.
- `hook_debug.jsonl` gets a `Stop` record with `"saved": true`.

## Encoding Notes

The Windows commands set:

```text
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

This prevents mojibake such as `You...e` replacing `You're`.

## Why Not Plugin Install?

RagMemory previously tested a Codex plugin package, but direct TOML hooks were more reliable in VS Code Codex.

Current recommended setup:

```text
direct hooks in %USERPROFILE%\.codex\config.toml
```

not:

```text
plugin marketplace install
```

## Remove

Delete the `[[hooks.UserPromptSubmit]]` and `[[hooks.Stop]]` blocks from:

```text
%USERPROFILE%\.codex\config.toml
```

Then restart Codex.
