"""Configure RagMemory Obsidian graph color groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_VAULT = Path("./.data/obsidian_memory")

COLOR_GROUPS = [
    ("path:\"active/messages\"", 0x00C8F0),
    ("path:\"active/structured\"", 0x009688),
    ("path:\"topics\"", 0x9966E6),
    ("path:\"files\"", 0xF36C00),
    ("path:\"profile\"", 0xE44DAD),
    ("path:\"forgotten\"", 0xB73636),
]


def configure_graph(vault: Path) -> Path:
    graph_path = vault / ".obsidian" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    if graph_path.exists():
        data = json.loads(graph_path.read_text(encoding="utf-8-sig"))
    else:
        data = {}

    data.setdefault("collapse-filter", True)
    data.setdefault("search", "")
    data.setdefault("showTags", False)
    data.setdefault("showAttachments", False)
    data.setdefault("hideUnresolved", False)
    data.setdefault("showOrphans", True)
    data["collapse-color-groups"] = False
    data["colorGroups"] = [
        {"query": query, "color": {"a": 1, "rgb": rgb}}
        for query, rgb in COLOR_GROUPS
    ]

    graph_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return graph_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set Obsidian graph colors for RagMemory.")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    args = parser.parse_args(argv)
    graph_path = configure_graph(args.vault)
    print(f"Configured Obsidian graph groups: {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
