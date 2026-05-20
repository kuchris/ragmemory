import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from openai import OpenAI

from ragmemory.memory import MemoryStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# MODEL = os.environ.get("LMSTUDIO_MODEL", "nemotron3-nano-4b-uncensored-hauhaucs-aggressive")
# MODEL = os.environ.get("LMSTUDIO_MODEL", "qwopus3.6-35b-a3b-v1")
MODEL = os.environ.get("LMSTUDIO_MODEL", "gemma-4-e4b-uncensored-hauhaucs-aggressive")
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
MAX_TOKENS = int(os.environ.get("LMSTUDIO_MAX_TOKENS", "10000"))

NOVEL_DB_PATH = os.environ.get("NOVEL_DB_PATH", "./chroma_novel")
CHAT_DB_PATH = os.environ.get("NOVEL_CHAT_DB_PATH", "./chroma_novel_chat")
GRAPH_EDGE_FILE = "graph_edges.jsonl"
NOVEL_TOP_K = 5
GRAPH_NEIGHBORS_PER_SIDE = 3
GRAPH_CHARACTER_NEIGHBORS = 2   # character-edge neighbours per retrieved chunk

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

novel_memory = MemoryStore(db_path=NOVEL_DB_PATH)
chat_memory = MemoryStore(db_path=CHAT_DB_PATH)


class StoryGraph:
    def __init__(self, path: Path):
        self.by_source: dict[str, dict[str, list[str]]] = {}
        self._load(path)

    def _load(self, path: Path):
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            edge = json.loads(line)
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if not source or not target or edge_type not in {"prev", "next", "character"}:
                continue
            self.by_source.setdefault(source, {}).setdefault(edge_type, []).append(target)

    def around(self, chunk_id: str) -> list[str]:
        nbrs = self.by_source.get(chunk_id, {})
        prev_ids      = nbrs.get("prev",      [])[:GRAPH_NEIGHBORS_PER_SIDE]
        next_ids      = nbrs.get("next",      [])[:GRAPH_NEIGHBORS_PER_SIDE]
        character_ids = nbrs.get("character", [])[:GRAPH_CHARACTER_NEIGHBORS]
        return list(reversed(prev_ids)) + [chunk_id] + next_ids + character_ids


story_graph = StoryGraph(Path(NOVEL_DB_PATH) / GRAPH_EDGE_FILE)


def fetch_chunks(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    fetched = novel_memory.collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = {
        chunk_id: {"id": chunk_id, "text": doc, "meta": meta}
        for chunk_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def build_novel_context(query: str) -> str:
    retrieved = novel_memory.retrieve(query, top_k=NOVEL_TOP_K)
    ordered_ids = []
    seen = set()

    for chunk in retrieved:
        for chunk_id in story_graph.around(chunk.id):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered_ids.append(chunk_id)

    chunks = fetch_chunks(ordered_ids)
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        meta = chunk["meta"]
        chapter = meta.get("chapter", "unknown chapter")
        chunk_index = meta.get("chunk_index", meta.get("message_id", "?"))
        parts.append(
            f"[chapter={chapter} | chunk_index={chunk_index}]\n{chunk['text']}"
        )
    return "=== Novel Context ===\n" + "\n---\n".join(parts)


def build_context(user_message: str) -> str:
    parts = []
    novel_context = build_novel_context(user_message)
    chat_context = chat_memory.build_context(user_message)

    if novel_context:
        parts.append(novel_context)
    if chat_context:
        parts.append(chat_context)
    return "\n\n".join(parts)


def call_model(context: str, user_message: str) -> str:
    system = (
        "You are a helpful assistant for discussing a novel. "
        "Use the provided novel context and conversation memory. "
        "If the answer is not supported by the context, say so clearly."
    )
    if context:
        system += f"\n\n{context}"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    parts = []
    hit_limit = False
    print("Assistant: ", end="", flush=True)
    with httpx.Client(timeout=None) as http_client:
        with http_client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                choice = event["choices"][0]
                delta = choice.get("delta", {})
                content_delta = delta.get("content") or ""
                if content_delta:
                    print(content_delta, end="", flush=True)
                    parts.append(content_delta)
                if choice.get("finish_reason") == "length":
                    hit_limit = True

    content = "".join(parts)
    if hit_limit:
        note = "\n\n[Note: response hit max token limit. Ask `continue` or increase LMSTUDIO_MAX_TOKENS.]"
        print(note, end="", flush=True)
        content += note
    print("\n")
    return content


def chat():
    print(
        "Novel Chat"
        f"  |  model: {MODEL}"
        f"  |  max tokens: {MAX_TOKENS}"
        f"  |  novel db: {NOVEL_DB_PATH}"
        f"  |  chat db: {CHAT_DB_PATH}"
        "  |  type 'quit' to exit\n"
    )
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        context = build_context(user_input)
        response = call_model(context, user_input)

        chat_memory.add_message("user", user_input, extract_structured=False)
        chat_memory.add_message("assistant", response, extract_structured=False)


if __name__ == "__main__":
    chat()
