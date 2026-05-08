# Complete Plan: Temporary Chat RAG + Recoverable Compression

## 🎯 Goals

- Preserve chat memory quality ("the chat remembers")
- Reduce main-model prefill latency
- Ensure no information is permanently lost
- Keep per-turn overhead low and predictable
- Scale to long conversations

---

## 🧠 Core Principles

**Memory ≠ Prompt**
The main model does not need to see all memory every turn. Memory must be stored, indexed, and retrievable, even if not included.

**Compression must be recoverable**
Removed content is archived, not deleted. RAG is the safety net.

**Incremental, not global**
Only process new content per turn. Never rechunk the full chat synchronously.

**Fast path first, smart path later**
Cheap extraction now. Better semantic consolidation in background.

---

## 🧩 High-Level Architecture

```
Raw Chat Log (source of truth)
        ↓
Incremental Semantic Chunking
        ↓
Temporary Per-Chat RAG Memory
        ↓
Hybrid Retrieval (per turn)
        ↓
Draft-Model Extractive Compression
        ↓
Main Model Prompt
        ↓
Answer
        ↓
Background Memory Consolidation
```

---

## 📦 Memory Layers

### Tier 1 — Recent Raw Window (Hot)
- Last 6–12 messages
- Included verbatim
- No compression
- Zero retrieval cost

### Tier 2 — Structured Memory (Warm)
High-value, compact facts:
- Decisions
- Constraints
- Preferences
- Technical conclusions
- Open questions

Example:
```json
{
  "type": "decision",
  "content": "Use temporary RAG memory to preserve compressed chat context",
  "importance": 0.97
}
```
Always included unless extremely long.

### Tier 3 — Retrieved Semantic Memory (Warm)
- Retrieved via hybrid RAG
- Candidate context for the prompt
- May be compressed

### Tier 4 — Archived / Removed Memory (Cold)
- Content removed by compression
- Fully indexed
- Only retrieved on demand
- **This tier prevents memory loss.**

---

## 🧠 Semantic Memory Objects (Chunking Strategy)

Do not chunk only by token count. Preferred object types:

- `preference`
- `decision`
- `constraint`
- `technical_concept`
- `implementation_plan`
- `open_question`
- `exact_quote`
- `code_reference`
- `removed_context_record`

Example object:
```json
{
  "type": "implementation_plan",
  "summary": "Use draft model to compress prompt after RAG retrieval",
  "source_text": "...",
  "tags": ["RAG", "compression", "draft-model"],
  "importance": 0.92,
  "turn_range": [21, 22]
}
```

---

## ⚡ Per-Turn Runtime Pipeline (Fast Path)

### Step 1 — Append Raw Messages
```python
raw_log.append(user_message)
```

### Step 2 — Retrieve Memory (Before Compression)
Hybrid retrieval using:
- Embeddings
- Keywords (BM25)
- Metadata/tags
- Recency boost
- Importance score

```python
retrieved = RAG.retrieve(query=user_message, top_k=10–20)
```

### Step 3 — Build Candidate Context
```python
candidate_context = (
    structured_memory
    + retrieved_memory
    + recent_raw_messages
    + current_user_message
)
```

### Step 4 — Draft-Model Extractive Compression
Only if context is too long.

Rules:
- ✅ Extractive only (no rewriting)
- ❌ Never remove:
  - Current user message
  - Recent raw turns
  - Decisions / constraints
  - Numbers, file names, code blocks

Output:
```json
{
  "kept": [...],
  "removed": [...]
}
```

### Step 5 — Removal Ledger Update (Critical)
Every removed chunk is logged:
```json
{
  "chunk_id": "mem_118",
  "summary": "Earlier PFlash keep_ratio discussion",
  "keywords": ["PFlash", "keep_ratio"],
  "importance": 0.88,
  "included_in_prompt": false
}
```
This makes compression recoverable.

### Step 6 — Main Model Call
Only sees:
```
kept_context + current_user_message
```

### Step 7 — Answer Returned
User sees a fast, coherent reply.

---

## 🧵 Background Pipeline (Slow Path, Async)

Runs after the response. Tasks:
- LLM-based memory consolidation
- Merge duplicate memories
- Improve summaries
- Create relations between memory objects
- Downgrade old raw chunks to cold memory

This never blocks the user.

---

## 🔍 Retrieval-on-Missing-Context Loop

Detect phrases like:
- "as we said earlier"
- "the thing we removed"
- "continue that design"
- "what did we decide?"

Then:
1. Retrieve from archived memory
2. Rebuild context
3. Answer with expanded memory

Optionally allow the model to emit:
```json
{ "action": "retrieve_memory", "query": "keep_ratio discussion" }
```

---

## ⏱️ Performance Expectations

### Normal Chat Turn (200–500 tokens)
| Operation | Time |
|-----------|------|
| Chunking + embedding | 5–30 ms |
| Retrieval | <10 ms |
| Draft compression | 20–100 ms |
> ✅ Negligible compared to main model

### Large Paste (5K–20K tokens)
| Operation | Time |
|-----------|------|
| Fast chunking | 100–500 ms |
| LLM extraction | background only |
> ✅ Main model still fast

---

## 🛡️ Safety Rules (Non-Negotiable)

Never aggressively compress:
- Code
- Legal / medical / financial content
- Explicit requirements
- Numbers, IDs, versions
- Tool schemas
- Commands
- File paths / function names

Use `keep_ratio ↑` or `compression = OFF` for these.

---

## 🧪 Why This Works

| Risk | Mitigation |
|------|------------|
| Compression removes important info | Archived memory + RAG |
| Slow chunking | Incremental + async |
| Chat feels forgetful | Structured + retrieved memory |
| Prompt too long | Draft compression |
| Wrong compression | Recoverable removal ledger |

---

## ✅ Final Mental Model

```
Prompt     = working set
Memory     = long-term store
Compression = view optimization
RAG        = recall mechanism
```

> You are not deleting memory — you are **paging it in and out intelligently**.

---

## Next Steps

- [ ] Design exact data schemas
- [ ] Sketch pseudo-code / APIs
- [ ] Compare this with PFlash-only
- [ ] Estimate GPU vs CPU costs
- [ ] Map into llama.cpp / vLLM / custom runtime
