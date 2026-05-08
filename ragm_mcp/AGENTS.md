# RAG Memory Instructions

You have access to a RAG memory system via MCP tools (rag-memory).

Every turn, two calls:

**Step 1 — Before answering:** `recall(user_message=<exact user message>)`
Returns relevant memory. Use it to inform your answer.

**Step 2 — After answering:** `save(summary=<1-2 sentence summary of your response>)`
Store a brief summary only — not the full response.

For large documents or long pastes: use `remember_document(text=<content>)`.

Never skip either step.
