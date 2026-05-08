"""
Build a character co-occurrence visualization from character_cache.json.

This does not call an LLM and does not rebuild character edges. It only reads:
    chroma_novel/character_cache.json

Run:
    uv run python visualize_characters.py
"""
import collections
import itertools
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DB_PATH = Path("./chroma_novel")
CACHE_PATH = DB_PATH / "character_cache.json"
OUT_PATH = Path("character_visualization.html")
TOP_N = 90
MIN_LINK = 3


def clean_names(names: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for raw in names:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return cleaned


cache: dict[str, list[str]] = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

character_counts: collections.Counter[str] = collections.Counter()
cooccurrence: collections.Counter[tuple[str, str]] = collections.Counter()

for names in cache.values():
    chars = sorted(clean_names(names))
    character_counts.update(chars)
    for a, b in itertools.combinations(chars, 2):
        cooccurrence[(a, b)] += 1

top_characters = [name for name, _ in character_counts.most_common(TOP_N)]
top_set = set(top_characters)

nodes = [
    {"id": name, "count": character_counts[name]}
    for name in top_characters
]

links = [
    {"source": a, "target": b, "value": value}
    for (a, b), value in cooccurrence.items()
    if a in top_set and b in top_set and value >= MIN_LINK
]

stats = {
    "chunks": len(cache),
    "nonemptyChunks": sum(1 for names in cache.values() if names),
    "characters": len(character_counts),
    "shownCharacters": len(nodes),
    "cooccurrencePairs": len(cooccurrence),
    "shownLinks": len(links),
    "minLink": MIN_LINK,
}

html = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Character Co-occurrence Network</title>
<link rel="icon" href="data:,">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {
    --bg: #0d1110;
    --panel: rgba(18, 24, 22, 0.90);
    --line: #2f3b36;
    --ink: #eef3ec;
    --muted: #95a59d;
    --accent: #d8b45d;
    --cyan: #6cc7bd;
    --rose: #e17663;
    --green: #97c969;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    height: 100vh;
    overflow: hidden;
    color: var(--ink);
    background:
      radial-gradient(circle at 20% 10%, rgba(108,199,189,0.16), transparent 30rem),
      radial-gradient(circle at 90% 22%, rgba(225,118,99,0.13), transparent 34rem),
      linear-gradient(135deg, rgba(216,180,93,0.08) 0 1px, transparent 1px 18px),
      var(--bg);
    font-family: "Georgia", "Noto Serif TC", "Times New Roman", serif;
  }

  #network { width: 100vw; height: 100vh; display: block; }

  .topbar {
    position: fixed;
    top: 18px;
    left: 18px;
    right: 18px;
    z-index: 10;
    display: grid;
    grid-template-columns: minmax(320px, 1fr) auto;
    gap: 18px;
    pointer-events: none;
  }

  .title, .control-panel, .side-panel {
    border: 1px solid var(--line);
    background: var(--panel);
    box-shadow: 0 22px 60px rgba(0,0,0,0.36);
    backdrop-filter: blur(12px);
    pointer-events: auto;
  }

  .title {
    padding: 18px 20px;
  }

  .eyebrow {
    color: var(--accent);
    font: 700 11px/1.2 "Consolas", "Courier New", monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }

  h1 {
    margin: 7px 0 0;
    font-size: clamp(28px, 4vw, 56px);
    line-height: 0.95;
    letter-spacing: 0;
  }

  .subtitle {
    margin-top: 10px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(90px, 1fr));
    gap: 8px;
    min-width: 480px;
  }

  .metric {
    border: 1px solid var(--line);
    background: rgba(8, 12, 11, 0.72);
    padding: 13px;
  }

  .metric b {
    display: block;
    font: 700 22px/1 "Consolas", "Courier New", monospace;
  }

  .metric span {
    display: block;
    margin-top: 7px;
    color: var(--muted);
    font: 700 10px/1.1 "Consolas", "Courier New", monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .control-panel {
    position: fixed;
    left: 18px;
    bottom: 18px;
    z-index: 10;
    width: 340px;
    padding: 16px;
  }

  label {
    display: block;
    margin-bottom: 13px;
    color: var(--muted);
    font: 700 11px/1.2 "Consolas", "Courier New", monospace;
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }

  input[type="range"], input[type="search"] {
    width: 100%;
    margin-top: 8px;
  }

  input[type="search"] {
    border: 1px solid var(--line);
    background: #0a0e0d;
    color: var(--ink);
    padding: 10px;
    font: 14px/1 "Consolas", "Courier New", monospace;
    outline: none;
  }

  input[type="search"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(216,180,93,0.16);
  }

  .checks {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
  }

  .checks label {
    margin: 0;
    text-transform: none;
    letter-spacing: 0;
    font-size: 13px;
    cursor: pointer;
  }

  .side-panel {
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 10;
    width: 360px;
    max-height: 48vh;
    padding: 16px;
    overflow: auto;
  }

  .side-panel h2 {
    margin: 0 0 10px;
    font-size: 18px;
  }

  .side-panel p, .side-panel li {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .side-panel ol {
    margin: 0;
    padding-left: 20px;
  }

  .node circle {
    stroke: rgba(238,243,236,0.78);
    stroke-width: 1.2;
    cursor: pointer;
  }

  .node text {
    fill: var(--ink);
    paint-order: stroke;
    stroke: rgba(13,17,16,0.88);
    stroke-width: 4px;
    font-size: 12px;
    pointer-events: none;
  }

  .link {
    stroke: rgba(148, 181, 172, 0.42);
    mix-blend-mode: screen;
  }

  .dimmed { opacity: 0.08 !important; }
  .hidden { display: none !important; }

  .tooltip {
    position: fixed;
    z-index: 30;
    pointer-events: none;
    opacity: 0;
    max-width: 340px;
    border: 1px solid var(--accent);
    background: #0a0e0d;
    color: var(--ink);
    padding: 12px 14px;
    box-shadow: 0 18px 44px rgba(0,0,0,0.45);
    font-size: 13px;
    line-height: 1.45;
  }

  .tooltip b { color: var(--accent); }

  @media (max-width: 980px) {
    body { overflow: auto; }
    #network { height: 900px; }
    .topbar { position: static; grid-template-columns: 1fr; padding: 14px; }
    .metrics { min-width: 0; grid-template-columns: repeat(2, 1fr); }
    .control-panel, .side-panel { position: static; width: auto; margin: 14px; }
  }
</style>
</head>
<body>
<svg id="network"></svg>

<div class="topbar">
  <section class="title">
    <div class="eyebrow">Character Graph / Co-occurrence</div>
    <h1>角色共現網絡</h1>
    <div class="subtitle">角色是節點；兩個角色出現在同一個 chunk，就累加一條共現邊。資料只來自 character_cache.json。</div>
  </section>
  <section class="metrics" id="metrics"></section>
</div>

<section class="control-panel">
  <label>連結強度下限：<span id="thresholdLabel"></span>
    <input id="threshold" type="range" min="1" max="80" value="3">
  </label>
  <label>顯示角色上限：<span id="topLabel"></span>
    <input id="topN" type="range" min="20" max="90" value="70">
  </label>
  <label>搜尋角色
    <input id="search" type="search" placeholder="例如：綾小路 / 堀北 / 龍園">
  </label>
  <div class="checks">
    <label><input id="labels" type="checkbox" checked> 顯示名字</label>
    <label><input id="weakLinks" type="checkbox" checked> 顯示弱連結</label>
  </div>
</section>

<section class="side-panel">
  <h2 id="sideTitle">Top co-occurrence</h2>
  <div id="sideContent"></div>
</section>

<div class="tooltip" id="tooltip"></div>

<script>
const rawNodes = __NODES__;
const rawLinks = __LINKS__;
const stats = __STATS__;

const state = { selected: null };
const svg = d3.select("#network");
const layer = svg.append("g");
const linkLayer = layer.append("g");
const nodeLayer = layer.append("g");

const els = {
  metrics: document.getElementById("metrics"),
  threshold: document.getElementById("threshold"),
  thresholdLabel: document.getElementById("thresholdLabel"),
  topN: document.getElementById("topN"),
  topLabel: document.getElementById("topLabel"),
  labels: document.getElementById("labels"),
  weakLinks: document.getElementById("weakLinks"),
  search: document.getElementById("search"),
  sideTitle: document.getElementById("sideTitle"),
  sideContent: document.getElementById("sideContent"),
  tooltip: document.getElementById("tooltip"),
};

const fmt = new Intl.NumberFormat("en-US");
const color = d3.scaleSequential()
  .domain([0, d3.max(rawNodes, d => d.count)])
  .interpolator(d3.interpolateMagma);
const radius = d3.scaleSqrt()
  .domain([0, d3.max(rawNodes, d => d.count)])
  .range([5, 34]);

svg.call(d3.zoom().scaleExtent([0.18, 5]).on("zoom", event => {
  layer.attr("transform", event.transform);
}));

let simulation = null;

function renderMetrics(nodes, links) {
  els.metrics.innerHTML = [
    ["Chunks", stats.chunks],
    ["Cache hits", stats.nonemptyChunks],
    ["Characters", nodes.length],
    ["Links", links.length],
  ].map(([label, value]) => `<div class="metric"><b>${fmt.format(value)}</b><span>${label}</span></div>`).join("");
}

function filteredGraph() {
  const threshold = Number(els.threshold.value);
  const topN = Number(els.topN.value);
  const query = els.search.value.trim().toLowerCase();
  const topSet = new Set(rawNodes.slice(0, topN).map(d => d.id));
  const nodes = rawNodes.filter(d => topSet.has(d.id) && (!query || d.id.toLowerCase().includes(query)));
  const nodeSet = new Set(nodes.map(d => d.id));
  const links = rawLinks.filter(d =>
    d.value >= threshold &&
    nodeSet.has(d.source) &&
    nodeSet.has(d.target)
  );
  const linked = new Set(links.flatMap(d => [d.source, d.target]));
  const activeNodes = nodes.filter(d => linked.has(d.id) || query);
  return { nodes: activeNodes, links };
}

function render() {
  const threshold = Number(els.threshold.value);
  const topN = Number(els.topN.value);
  els.thresholdLabel.textContent = threshold;
  els.topLabel.textContent = topN;

  const { nodes, links } = filteredGraph();
  renderMetrics(nodes, links);
  renderSidePanel(nodes, links);

  if (simulation) simulation.stop();
  linkLayer.selectAll("*").remove();
  nodeLayer.selectAll("*").remove();

  const width = window.innerWidth;
  const height = window.innerHeight;
  svg.attr("viewBox", [0, 0, width, height]);

  const nodeById = new Map(nodes.map(d => [d.id, { ...d }]));
  const graphLinks = links
    .map(d => ({ ...d, source: nodeById.get(d.source), target: nodeById.get(d.target) }))
    .filter(d => d.source && d.target);
  const graphNodes = [...nodeById.values()];

  simulation = d3.forceSimulation(graphNodes)
    .force("link", d3.forceLink(graphLinks).id(d => d.id).distance(d => Math.max(45, 190 / Math.log(d.value + 2))).strength(d => Math.min(0.9, 0.18 + d.value / 80)))
    .force("charge", d3.forceManyBody().strength(d => -180 - radius(d.count) * 8))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(d => radius(d.count) + 8));

  const link = linkLayer.selectAll("line")
    .data(graphLinks)
    .join("line")
    .attr("class", d => `link ${!els.weakLinks.checked && d.value < threshold * 2 ? "hidden" : ""}`)
    .attr("stroke-width", d => Math.max(1, Math.sqrt(d.value) * 0.85));

  const node = nodeLayer.selectAll("g")
    .data(graphNodes)
    .join("g")
    .attr("class", "node")
    .call(d3.drag()
      .on("start", dragStart)
      .on("drag", dragged)
      .on("end", dragEnd));

  node.append("circle")
    .attr("r", d => radius(d.count))
    .attr("fill", d => color(d.count))
    .on("mousemove", (event, d) => showTip(event, nodeTip(d, graphLinks)))
    .on("mouseleave", hideTip)
    .on("click", (event, d) => selectNode(d, node, link, graphLinks));

  node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", d => -radius(d.count) - 7)
    .text(d => d.id)
    .style("display", els.labels.checked ? null : "none");

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

function renderSidePanel(nodes, links) {
  const sorted = [...links].sort((a, b) => b.value - a.value).slice(0, 14);
  els.sideTitle.textContent = "Top co-occurrence";
  if (!sorted.length) {
    els.sideContent.innerHTML = "<p>目前篩選沒有共現連結。</p>";
    return;
  }
  els.sideContent.innerHTML = `<ol>${sorted.map(d =>
    `<li><b>${escapeHtml(d.source)}</b> - <b>${escapeHtml(d.target)}</b>：${fmt.format(d.value)} chunks</li>`
  ).join("")}</ol>`;
}

function nodeTip(d, links) {
  const related = links
    .filter(l => l.source.id === d.id || l.target.id === d.id)
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)
    .map(l => {
      const other = l.source.id === d.id ? l.target.id : l.source.id;
      return `${escapeHtml(other)} (${fmt.format(l.value)})`;
    });
  return `<b>${escapeHtml(d.id)}</b><br>出現 chunks：${fmt.format(d.count)}<br>常一起出現：${related.join(", ") || "無"}`;
}

function selectNode(d, node, link, links) {
  if (state.selected === d.id) {
    state.selected = null;
    node.classed("dimmed", false);
    link.classed("dimmed", false);
    return;
  }
  state.selected = d.id;
  const connected = new Set([d.id]);
  links.forEach(l => {
    if (l.source.id === d.id) connected.add(l.target.id);
    if (l.target.id === d.id) connected.add(l.source.id);
  });
  node.classed("dimmed", nd => !connected.has(nd.id));
  link.classed("dimmed", l => l.source.id !== d.id && l.target.id !== d.id);
}

function showTip(event, html) {
  els.tooltip.innerHTML = html;
  els.tooltip.style.opacity = 1;
  els.tooltip.style.left = `${event.clientX + 14}px`;
  els.tooltip.style.top = `${event.clientY - 10}px`;
}

function hideTip() {
  els.tooltip.style.opacity = 0;
}

function dragStart(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

function dragEnd(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

[els.threshold, els.topN, els.labels, els.weakLinks, els.search].forEach(el => {
  el.addEventListener("input", render);
});
window.addEventListener("resize", render);

render();
</script>
</body>
</html>
"""

OUT_PATH.write_text(
    html
    .replace("__NODES__", json.dumps(nodes, ensure_ascii=True))
    .replace("__LINKS__", json.dumps(links, ensure_ascii=True))
    .replace("__STATS__", json.dumps(stats, ensure_ascii=True)),
    encoding="utf-8",
)

print(f"Read cache: {CACHE_PATH}")
print(f"Chunks: {stats['chunks']} | nonempty: {stats['nonemptyChunks']}")
print(f"Characters: {stats['characters']} | shown: {stats['shownCharacters']}")
print(f"Co-occurrence pairs: {stats['cooccurrencePairs']} | shown links: {stats['shownLinks']}")
print(f"Done: {OUT_PATH}")
