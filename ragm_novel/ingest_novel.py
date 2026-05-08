"""
Ingest a plain-text novel into RagMemory chunks.

Run:
    uv run python ingest_novel.py path\to\novel.txt

Use the same DB as chat.py:
    uv run python ingest_novel.py path\to\novel.txt --db-path ./chroma_structured_test
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import uuid
from pathlib import Path

from memory import MemoryStore, score_importance

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DB_PATH = "./chroma_novel"
DEFAULT_TARGET_CHARS = 900
DEFAULT_OVERLAP_CHARS = 120
GRAPH_EDGE_FILE = "graph_edges.jsonl"

CHAPTER_RE = re.compile(
    r"^\s*(\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u96f6\u3007\u4e24\d]+[\u7ae0\u8282\u56de\u5377\u90e8].*|chapter\s+\d+.*)\s*$",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^\u3002\uff01\uff1f.!?]+[\u3002\uff01\uff1f.!?]*")


def progress(iterable, total: int, desc: str, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode {path}")


def split_chapters(text: str) -> list[tuple[str, str]]:
    chapters = []
    current_title = "Front Matter"
    current_lines = []

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if CHAPTER_RE.match(line.strip()):
            if current_lines:
                chapters.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        chapters.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in chapters if body]


def split_long_paragraph(paragraph: str, target_chars: int) -> list[str]:
    sentences = [m.group(0).strip() for m in SENTENCE_RE.finditer(paragraph) if m.group(0).strip()]
    if not sentences:
        return [paragraph[i : i + target_chars] for i in range(0, len(paragraph), target_chars)]

    chunks = []
    current = ""
    for sentence in sentences:
        candidate = current + sentence
        if len(candidate) > target_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_chapter(
    title: str,
    body: str,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    base_chunks = []
    current = ""

    for paragraph in paragraphs:
        parts = (
            split_long_paragraph(paragraph, target_chars)
            if len(paragraph) > target_chars
            else [paragraph]
        )
        for part in parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) > target_chars and current:
                base_chunks.append(current.strip())
                current = part
            else:
                current = candidate

    if current.strip():
        base_chunks.append(current.strip())

    chunks = []
    previous_tail = ""
    for chunk in base_chunks:
        text = f"[Chapter: {title}]\n\n{chunk}"
        if previous_tail:
            text = f"[Chapter: {title}]\n[Previous overlap]\n{previous_tail}\n\n{chunk}"
        chunks.append(text)
        previous_tail = chunk[-overlap_chars:] if overlap_chars > 0 else ""

    return chunks


def ingest(
    novel_path: Path,
    db_path: str,
    title: str,
    target_chars: int,
    overlap_chars: int,
    dry_run: bool,
    reset_db: bool,
) -> int:
    text = read_text(novel_path)
    chapters = split_chapters(text)

    documents = []
    metadatas = []
    ids = []
    graph_edges = []
    chunk_index = 0

    for chapter_title, chapter_body in progress(
        chapters,
        total=len(chapters),
        desc="Chunking story",
        unit="chapter",
    ):
        chapter_ids = []
        for chunk in chunk_chapter(chapter_title, chapter_body, target_chars, overlap_chars):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            chapter_ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "message_id": chunk_index,
                    "role": "novel",
                    "importance": score_importance(chunk),
                    "source_title": title,
                    "chapter": chapter_title,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
        for left, right in zip(chapter_ids, chapter_ids[1:]):
            graph_edges.append(
                {
                    "source": left,
                    "target": right,
                    "type": "next",
                    "weight": 1.0,
                    "source_title": title,
                    "chapter": chapter_title,
                }
            )
            graph_edges.append(
                {
                    "source": right,
                    "target": left,
                    "type": "prev",
                    "weight": 1.0,
                    "source_title": title,
                    "chapter": chapter_title,
                }
            )

    print(f"Novel: {title}")
    print(f"Chapters: {len(chapters)}")
    print(f"Chunks: {len(documents)}")
    print(f"Graph edges: {len(graph_edges)}")
    if documents:
        lengths = [len(doc) for doc in documents]
        print(f"Chunk chars: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) // len(lengths)}")

    if dry_run:
        return len(documents)

    if reset_db:
        target = Path(db_path).resolve()
        cwd = Path.cwd().resolve()
        if not target.is_relative_to(cwd):
            raise RuntimeError(f"Refusing to reset DB outside this repo: {target}")
        if target.exists():
            shutil.rmtree(target)

    store = MemoryStore(db_path=db_path)
    before = store.collection.count()

    batch_size = 100
    batch_starts = range(0, len(documents), batch_size)
    batch_count = (len(documents) + batch_size - 1) // batch_size
    for start in progress(
        batch_starts,
        total=batch_count,
        desc="Embedding/storing",
        unit="batch",
    ):
        end = start + batch_size
        store.collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    graph_path = store.db_path / GRAPH_EDGE_FILE
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("w", encoding="utf-8") as f:
        for edge in graph_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    after = store.collection.count()
    print(f"DB: {db_path}")
    print(f"Stored chunks: {after - before} | total: {after}")
    print(f"Stored graph edges: {len(graph_edges)} | file: {graph_path}")
    return len(documents)


def main():
    parser = argparse.ArgumentParser(description="Chunk and ingest a plain-text novel into RagMemory.")
    parser.add_argument("novel_path", type=Path)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--title", default=None)
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()

    title = args.title or args.novel_path.stem
    ingest(
        novel_path=args.novel_path,
        db_path=args.db_path,
        title=title,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        dry_run=args.dry_run,
        reset_db=args.reset_db,
    )


if __name__ == "__main__":
    main()
