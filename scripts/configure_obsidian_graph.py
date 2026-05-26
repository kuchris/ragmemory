"""Configure RagMemory Obsidian graph color groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_obsidian import OBSIDIAN_GRAPH_COLOR_GROUPS, configure_obsidian_graph


DEFAULT_VAULT = Path("./.data/obsidian_memory")
COLOR_GROUPS = OBSIDIAN_GRAPH_COLOR_GROUPS


def configure_graph(vault: Path) -> Path:
    return configure_obsidian_graph(vault)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set Obsidian graph colors for RagMemory.")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    args = parser.parse_args(argv)
    graph_path = configure_graph(args.vault)
    print(f"Configured Obsidian graph groups: {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
