"""
Render the generated Obsidian memory graph as a growing animated GIF.

Run:
    uv run python scripts/animate_obsidian_graph.py --obsidian ./.data/obsidian_memory --output ./.data/graph_animation/ragmemory-map-formed.gif
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path


DEFAULT_OBSIDIAN_PATH = Path("./.data/obsidian_memory")
DEFAULT_OUTPUT_PATH = Path("./.data/graph_animation/ragmemory-map-formed.gif")
WIKILINK_RE = re.compile(r"\[\[([^|\]#]+)")
MESSAGE_ID_RE = re.compile(r"message_id:\s*([0-9]+)")
MSG_STEM_RE = re.compile(r"msg-([0-9]+)")
PALETTE = [
    (255, 255, 255),  # 0 background
    (205, 211, 218),  # 1 edge
    (0, 150, 136),    # 2 structured
    (153, 102, 230),  # 3 topic
    (55, 130, 220),   # 4 topic group
    (0, 200, 240),    # 5 message
    (243, 108, 0),    # 6 file
    (228, 77, 173),   # 7 profile
    (70, 70, 70),     # 8 progress
]


class Note:
    def __init__(self, stem: str, note_type: str, message_id: int | None):
        self.stem = stem
        self.note_type = note_type
        self.message_id = message_id
        self.links: list[str] = []


def stable_float(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(0xFFFFFFFFFFFF)


def strip_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1], parts[2]


def note_type_for(relative_stem: str) -> str | None:
    if relative_stem.startswith("active/structured/"):
        return "structured"
    if relative_stem.startswith("forgotten/structured/"):
        return "structured"
    if relative_stem.startswith("topic_groups/"):
        return "topic_group"
    if relative_stem.startswith("topics/"):
        return "topic"
    if relative_stem.startswith("files/"):
        return "file"
    if relative_stem.startswith("profile/"):
        return "profile"
    if relative_stem.startswith("active/messages/"):
        return "message"
    if relative_stem.startswith("forgotten/messages/"):
        return "message"
    if relative_stem == "index" or relative_stem.startswith("maps/"):
        return "navigation"
    if "/" not in relative_stem:
        return "navigation"
    return None


def load_notes(root: Path, include_messages: bool, include_navigation: bool) -> dict[str, Note]:
    notes: dict[str, Note] = {}
    aliases: dict[str, str] = {}
    for path in root.rglob("*.md"):
        relative_stem = path.relative_to(root).with_suffix("").as_posix()
        note_type = note_type_for(relative_stem)
        if note_type is None or (note_type == "message" and not include_messages):
            continue
        if note_type == "navigation" and not include_navigation:
            continue
        text = path.read_text(encoding="utf-8-sig")
        frontmatter, body = strip_frontmatter(text)
        message_id = None
        if note_type == "message":
            match = MSG_STEM_RE.search(path.stem)
            message_id = int(match.group(1)) if match else None
        else:
            match = MESSAGE_ID_RE.search(frontmatter)
            message_id = int(match.group(1)) if match else None
        note = Note(relative_stem, note_type, message_id)
        note.links = [target.strip() for target in WIKILINK_RE.findall(body)]
        notes[relative_stem] = note
        aliases.setdefault(path.stem, relative_stem)

    for note in notes.values():
        resolved = []
        for target in note.links:
            if target in notes:
                resolved.append(target)
            elif target in aliases and aliases[target] in notes:
                resolved.append(aliases[target])
        note.links = sorted(set(resolved))
    return notes


def build_edges(notes: dict[str, Note]) -> set[tuple[str, str]]:
    edges = set()
    for source, note in notes.items():
        for target in note.links:
            if target in notes and source != target:
                edges.add(tuple(sorted((source, target))))
    return edges


def node_order(notes: dict[str, Note], edges: set[tuple[str, str]]) -> dict[str, int]:
    fallback = max((note.message_id or 0 for note in notes.values()), default=0) + 1
    order = {
        stem: note.message_id
        for stem, note in notes.items()
        if note.message_id is not None
    }
    for _ in range(4):
        changed = False
        for left, right in edges:
            left_order = order.get(left)
            right_order = order.get(right)
            if left_order is not None and right_order is None:
                order[right] = left_order
                changed = True
            elif right_order is not None and left_order is None:
                order[left] = right_order
                changed = True
        if not changed:
            break
    return {stem: order.get(stem, fallback) for stem in notes}


def trim_graph(
    notes: dict[str, Note],
    edges: set[tuple[str, str]],
    order: dict[str, int],
    max_nodes: int,
) -> tuple[dict[str, Note], set[tuple[str, str]], dict[str, int]]:
    if max_nodes <= 0 or len(notes) <= max_nodes:
        return notes, edges, order
    degree = {stem: 0 for stem in notes}
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    type_rank = {
        "topic_group": 0,
        "topic": 1,
        "profile": 2,
        "file": 3,
        "structured": 4,
        "message": 5,
        "navigation": 6,
    }
    ranked = sorted(
        notes,
        key=lambda stem: (type_rank.get(notes[stem].note_type, 9), -degree[stem], order[stem], stem),
    )
    keep = set(ranked[:max_nodes])
    trimmed_notes = {stem: note for stem, note in notes.items() if stem in keep}
    trimmed_edges = {edge for edge in edges if edge[0] in keep and edge[1] in keep}
    trimmed_order = {stem: value for stem, value in order.items() if stem in keep}
    return trimmed_notes, trimmed_edges, trimmed_order


def vector_angle(vectors: list[tuple[float, float]], fallback: float) -> float:
    if not vectors:
        return fallback
    x = sum(item[0] for item in vectors)
    y = sum(item[1] for item in vectors)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return fallback
    return math.atan2(y, x)


def layout_nodes(notes: dict[str, Note], edges: set[tuple[str, str]], width: int, height: int) -> dict[str, tuple[int, int]]:
    by_type: dict[str, list[str]] = {}
    for stem, note in notes.items():
        by_type.setdefault(note.note_type, []).append(stem)
    for stems in by_type.values():
        stems.sort()

    positions: dict[str, tuple[float, float]] = {}
    center_x = width / 2
    center_y = height / 2
    scale = min(width, height) * 0.44

    def place_ring(stems: list[str], radius: float, offset: float = 0.0) -> None:
        total = max(1, len(stems))
        for index, stem in enumerate(stems):
            angle = offset + (2 * math.pi * index / total)
            positions[stem] = (math.cos(angle) * radius, math.sin(angle) * radius)

    place_ring(by_type.get("topic_group", []), 0.14, -math.pi / 2)
    place_ring(by_type.get("topic", []), 0.55, 0.12)
    place_ring(by_type.get("file", []), 0.72, 0.35)
    place_ring(by_type.get("profile", []), 0.24, math.pi / 3)
    place_ring(by_type.get("navigation", []), 0.96, math.pi / 5)

    neighbors: dict[str, list[str]] = {stem: [] for stem in notes}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)

    for stem in by_type.get("structured", []):
        vectors = [positions[target] for target in neighbors[stem] if target in positions]
        fallback = stable_float(stem) * 2 * math.pi
        angle = vector_angle(vectors, fallback)
        radius = 0.78 + (stable_float(stem + ":r") - 0.5) * 0.12
        positions[stem] = (math.cos(angle) * radius, math.sin(angle) * radius)

    for stem in by_type.get("message", []):
        vectors = [positions[target] for target in neighbors[stem] if target in positions]
        fallback = stable_float(stem) * 2 * math.pi
        angle = vector_angle(vectors, fallback)
        radius = 0.92
        positions[stem] = (math.cos(angle) * radius, math.sin(angle) * radius)

    return {
        stem: (int(center_x + x * scale), int(center_y + y * scale))
        for stem, (x, y) in positions.items()
    }


def blank_canvas(width: int, height: int) -> bytearray:
    return bytearray([0]) * (width * height)


def set_pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color: int) -> None:
    if 0 <= x < width and 0 <= y < height:
        canvas[y * width + x] = color


def draw_line(canvas: bytearray, width: int, height: int, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        set_pixel(canvas, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(canvas: bytearray, width: int, height: int, center: tuple[int, int], radius: int, color: int) -> None:
    cx, cy = center
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                set_pixel(canvas, width, height, x, y, color)


def draw_progress(canvas: bytearray, width: int, height: int, fraction: float) -> None:
    bar_width = int((width - 40) * max(0.0, min(1.0, fraction)))
    y = height - 20
    for x in range(20, 20 + bar_width):
        for yy in range(y, y + 5):
            set_pixel(canvas, width, height, x, yy, 8)


def node_color(note_type: str) -> int:
    return {
        "structured": 2,
        "topic": 3,
        "topic_group": 4,
        "message": 5,
        "file": 6,
        "profile": 7,
        "navigation": 8,
    }.get(note_type, 2)


def node_radius(note_type: str) -> int:
    return {
        "topic_group": 7,
        "topic": 4,
        "file": 4,
        "profile": 5,
        "message": 2,
        "structured": 2,
        "navigation": 4,
    }.get(note_type, 2)


def render_frame(
    notes: dict[str, Note],
    edges: set[tuple[str, str]],
    positions: dict[str, tuple[int, int]],
    visible: set[str],
    width: int,
    height: int,
    fraction: float,
) -> bytes:
    canvas = blank_canvas(width, height)
    for left, right in sorted(edges):
        if left in visible and right in visible:
            draw_line(canvas, width, height, positions[left], positions[right], 1)
    for note_type in ("navigation", "message", "structured", "file", "profile", "topic", "topic_group"):
        for stem, note in notes.items():
            if stem in visible and note.note_type == note_type:
                draw_circle(canvas, width, height, positions[stem], node_radius(note_type), node_color(note_type))
    draw_progress(canvas, width, height, fraction)
    return bytes(canvas)


def lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    clear_interval = 10
    codes: list[int] = [clear_code]
    since_clear = 0
    for value in indices:
        if since_clear >= clear_interval:
            codes.append(clear_code)
            since_clear = 0
        codes.append(value)
        since_clear += 1
    codes.append(end_code)

    packed = bytearray()
    bit_buffer = 0
    bit_count = 0
    for code in codes:
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
    if bit_count:
        packed.append(bit_buffer & 0xFF)
    return bytes(packed)


def gif_subblocks(data: bytes) -> bytes:
    parts = bytearray()
    for index in range(0, len(data), 255):
        chunk = data[index:index + 255]
        parts.append(len(chunk))
        parts.extend(chunk)
    parts.append(0)
    return bytes(parts)


def write_gif(path: Path, frames: list[bytes], width: int, height: int, delay_cs: int, final_delay_cs: int) -> None:
    min_code_size = 4
    palette_size = 16
    palette = PALETTE + [(255, 255, 255)] * (palette_size - len(PALETTE))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"GIF89a")
        handle.write(width.to_bytes(2, "little"))
        handle.write(height.to_bytes(2, "little"))
        handle.write(bytes([0b10000011, 0, 0]))
        for red, green, blue in palette:
            handle.write(bytes([red, green, blue]))
        handle.write(b"!\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")
        for index, frame in enumerate(frames):
            delay = final_delay_cs if index == len(frames) - 1 else delay_cs
            handle.write(b"!\xF9\x04")
            handle.write(bytes([0b00000100]))
            handle.write(delay.to_bytes(2, "little"))
            handle.write(b"\x00\x00")
            handle.write(b",")
            handle.write((0).to_bytes(2, "little"))
            handle.write((0).to_bytes(2, "little"))
            handle.write(width.to_bytes(2, "little"))
            handle.write(height.to_bytes(2, "little"))
            handle.write(b"\x00")
            handle.write(bytes([min_code_size]))
            handle.write(gif_subblocks(lzw_encode(frame, min_code_size)))
        handle.write(b";")


def build_thresholds(order: dict[str, int], step: int, max_frames: int) -> list[int]:
    values = sorted(set(order.values()))
    if not values:
        return [0]
    start, end = min(values), max(values)
    thresholds = list(range(start, end + 1, max(1, step)))
    if thresholds[-1] != end:
        thresholds.append(end)
    if len(thresholds) > max_frames:
        stride = math.ceil(len(thresholds) / max_frames)
        thresholds = thresholds[::stride]
        if thresholds[-1] != end:
            thresholds.append(end)
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate the generated RagMemory Obsidian graph as a GIF.")
    parser.add_argument("--obsidian", type=Path, default=DEFAULT_OBSIDIAN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--step", type=int, default=25, help="Message-id interval revealed per frame.")
    parser.add_argument("--max-frames", type=int, default=70)
    parser.add_argument("--max-nodes", type=int, default=0, help="Use 0 to render every memory graph node.")
    parser.add_argument("--delay-ms", type=int, default=90)
    parser.add_argument("--final-delay-ms", type=int, default=1800)
    parser.add_argument("--include-messages", action="store_true", help="Deprecated: messages are included by default.")
    parser.add_argument("--exclude-messages", action="store_true", help="Hide raw message nodes for a smaller explainer GIF.")
    parser.add_argument("--include-navigation", action="store_true", help="Also render index/maps navigation notes.")
    args = parser.parse_args()

    include_messages = not args.exclude_messages
    notes = load_notes(
        args.obsidian,
        include_messages=include_messages,
        include_navigation=args.include_navigation,
    )
    if not notes:
        raise SystemExit(f"No graph notes found under {args.obsidian}")
    edges = build_edges(notes)
    order = node_order(notes, edges)
    notes, edges, order = trim_graph(notes, edges, order, args.max_nodes)
    positions = layout_nodes(notes, edges, args.width, args.height)
    thresholds = build_thresholds(order, args.step, args.max_frames)
    frames = []
    for index, threshold in enumerate(thresholds):
        visible = {stem for stem, value in order.items() if value <= threshold}
        fraction = 0 if len(thresholds) == 1 else index / (len(thresholds) - 1)
        frames.append(render_frame(notes, edges, positions, visible, args.width, args.height, fraction))
    frames.append(render_frame(notes, edges, positions, set(notes), args.width, args.height, 1.0))
    write_gif(
        args.output,
        frames,
        args.width,
        args.height,
        max(1, args.delay_ms // 10),
        max(1, args.final_delay_ms // 10),
    )
    print(
        f"Wrote {args.output} with {len(frames)} frame(s), "
        f"{len(notes)} node(s), {len(edges)} edge(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
