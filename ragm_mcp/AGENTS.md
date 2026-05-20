## RagMemory MCP

If the `rag-memory` MCP tools are available, use them for persistent memory.

At the start of each user turn:
1. Call `recall(user_message=<exact user message>)`.
2. Use relevant returned memory.
3. Ignore memory that is unrelated, stale, or contradicted by the current user.

At the end of each turn:
1. Call `save(summary=<brief summary>)`.
2. Save only durable facts, decisions, preferences, constraints, project state, or important outcomes.
3. Do not save the full assistant response.
4. Do not save secrets, API keys, passwords, tokens, or private credentials.

For long pasted docs/logs/code:
- Use `remember_document(text=<content>)`.

For removal:
1. Call `forget_preview(...)` first.
2. Show the user what would be forgotten.
3. Call `forget_confirm(...)` only after explicit user approval.

Use `memory_stats()` only for debugging or when the user asks about memory state.