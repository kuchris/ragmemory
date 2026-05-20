import collections
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EDGE_PATH = Path(".data/chroma_novel/graph_edges.jsonl")
OUT_PATH = Path("graph_visualization.html")


def volume_key(chapter: str) -> str:
    match = re.match(r"^(第\S+卷)", chapter)
    if match:
        return match.group(1)
    return "未分卷"


chapters = collections.OrderedDict()
with EDGE_PATH.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        edge = json.loads(line)
        key = (edge["source_title"], edge["chapter"])
        record = chapters.setdefault(
            key,
            {
                "title": edge["source_title"],
                "chapter": edge["chapter"],
                "volume": volume_key(edge["chapter"]),
                "nodes": set(),
                "nextEdges": 0,
                "prevEdges": 0,
            },
        )
        record["nodes"].add(edge["source"])
        record["nodes"].add(edge["target"])
        if edge["type"] == "next":
            record["nextEdges"] += 1
        elif edge["type"] == "prev":
            record["prevEdges"] += 1


titles = collections.OrderedDict()
for record in chapters.values():
    title = record["title"]
    if title not in titles:
        titles[title] = {
            "title": title,
            "chapters": [],
        }
    titles[title]["chapters"].append(
        {
            "chapter": record["chapter"],
            "volume": record["volume"],
            "chunks": len(record["nodes"]),
            "nextEdges": record["nextEdges"],
            "prevEdges": record["prevEdges"],
            "edges": record["nextEdges"] + record["prevEdges"],
        }
    )

data = list(titles.values())

html = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Memory - Story Graph</title>
<link rel="icon" href="data:,">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {
    --bg: #10100e;
    --panel: #181814;
    --panel-2: #202017;
    --ink: #f2ebdc;
    --muted: #aaa18e;
    --line: #3c382d;
    --gold: #d7a84d;
    --cyan: #72b7bb;
    --green: #92b66b;
    --rose: #d47766;
    --violet: #9a8fcb;
    --shadow: rgba(0, 0, 0, 0.38);
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    background:
      linear-gradient(90deg, rgba(215,168,77,0.05) 1px, transparent 1px),
      linear-gradient(180deg, rgba(215,168,77,0.04) 1px, transparent 1px),
      radial-gradient(circle at 18% 8%, rgba(114,183,187,0.12), transparent 28rem),
      radial-gradient(circle at 84% 16%, rgba(212,119,102,0.10), transparent 30rem),
      var(--bg);
    background-size: 32px 32px, 32px 32px, auto, auto, auto;
    color: var(--ink);
    font-family: "Georgia", "Times New Roman", "Noto Serif TC", serif;
  }

  .shell { max-width: 1500px; margin: 0 auto; padding: 28px; }

  header {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) auto;
    gap: 28px;
    align-items: end;
    padding: 18px 0 26px;
    border-bottom: 1px solid var(--line);
  }

  .eyebrow {
    color: var(--gold);
    font: 700 12px/1.2 "Consolas", "Courier New", monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }

  h1 {
    margin: 8px 0 0;
    max-width: 900px;
    font-size: clamp(34px, 5vw, 72px);
    line-height: 0.92;
    letter-spacing: 0;
  }

  .subtitle {
    margin: 16px 0 0;
    max-width: 760px;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.65;
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(110px, 1fr));
    gap: 10px;
    min-width: min(620px, 100%);
  }

  .metric {
    border: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012));
    padding: 14px;
    min-height: 88px;
  }

  .metric b {
    display: block;
    font: 700 26px/1 "Consolas", "Courier New", monospace;
    color: var(--ink);
  }

  .metric span {
    display: block;
    margin-top: 9px;
    color: var(--muted);
    font: 700 11px/1.2 "Consolas", "Courier New", monospace;
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }

  .layout {
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    gap: 22px;
    margin-top: 22px;
  }

  aside {
    align-self: start;
    position: sticky;
    top: 16px;
    border: 1px solid var(--line);
    background: rgba(24, 24, 20, 0.86);
    box-shadow: 0 18px 44px var(--shadow);
    padding: 18px;
  }

  label {
    display: block;
    margin: 0 0 14px;
    color: var(--muted);
    font: 700 11px/1.2 "Consolas", "Courier New", monospace;
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }

  select, input {
    width: 100%;
    margin-top: 7px;
    border: 1px solid var(--line);
    background: #0f100d;
    color: var(--ink);
    padding: 10px 11px;
    font: 14px/1.2 "Consolas", "Courier New", monospace;
    outline: none;
  }

  select:focus, input:focus { border-color: var(--gold); box-shadow: 0 0 0 2px rgba(215,168,77,0.18); }

  .segmented {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px;
    padding: 4px;
    background: #0f100d;
    border: 1px solid var(--line);
  }

  .segmented button {
    border: 0;
    background: transparent;
    color: var(--muted);
    padding: 9px 7px;
    cursor: pointer;
    font: 700 12px/1 "Consolas", "Courier New", monospace;
  }

  .segmented button.active {
    background: var(--gold);
    color: #17130a;
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
    margin-top: 14px;
  }

  main { min-width: 0; }

  .panel {
    border: 1px solid var(--line);
    background: rgba(24, 24, 20, 0.9);
    box-shadow: 0 18px 44px var(--shadow);
    margin-bottom: 18px;
  }

  .panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    padding: 16px 18px;
    border-bottom: 1px solid var(--line);
  }

  .panel-head h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
  }

  .panel-head span {
    color: var(--muted);
    font: 12px/1.2 "Consolas", "Courier New", monospace;
  }

  #chart { overflow-x: auto; min-height: 520px; }
  svg { display: block; }
  .axis text { fill: var(--muted); font-family: "Consolas", "Courier New", monospace; }
  .axis path, .axis line { stroke: var(--line); }
  .grid line { stroke: rgba(170,161,142,0.16); }
  .grid path { display: none; }

  .tooltip {
    position: fixed;
    z-index: 20;
    pointer-events: none;
    opacity: 0;
    transform: translate(12px, -12px);
    max-width: 360px;
    border: 1px solid var(--gold);
    background: #10100e;
    color: var(--ink);
    padding: 12px 14px;
    box-shadow: 0 18px 40px var(--shadow);
    font-size: 13px;
    line-height: 1.5;
  }

  .tooltip b { color: var(--gold); }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }

  th, td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(170,161,142,0.16);
    text-align: left;
    vertical-align: top;
  }

  th {
    color: var(--gold);
    font: 700 11px/1.2 "Consolas", "Courier New", monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    cursor: pointer;
    user-select: none;
  }

  td.num {
    font-family: "Consolas", "Courier New", monospace;
    color: var(--cyan);
    white-space: nowrap;
  }

  tbody tr:hover { background: rgba(215,168,77,0.06); }
  .table-wrap { max-height: 520px; overflow: auto; }

  .empty {
    padding: 42px;
    color: var(--muted);
    text-align: center;
  }

  @media (max-width: 980px) {
    .shell { padding: 18px; }
    header { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .layout { grid-template-columns: 1fr; }
    aside { position: static; }
  }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <div class="eyebrow">RAG Memory / Story Graph</div>
      <h1>小說 chunk 結構視覺化</h1>
      <p class="subtitle">從 graph_edges.jsonl 聚合章節節點與 next / prev 邊，檢查哪些章節被切得過密、哪些卷的敘事跨度最大。</p>
    </div>
    <section class="metrics" id="metrics"></section>
  </header>

  <div class="layout">
    <aside>
      <label>視圖
        <div class="segmented" id="modeButtons">
          <button data-mode="bars" class="active">Density</button>
          <button data-mode="timeline">Spine</button>
          <button data-mode="table">Index</button>
        </div>
      </label>
      <label>書籍
        <select id="titleFilter"></select>
      </label>
      <label>分卷
        <select id="volumeFilter"></select>
      </label>
      <label>搜尋章節
        <input id="searchInput" type="search" placeholder="例如：游泳池 / 特典 / 第七卷">
      </label>
      <label>最少 chunks
        <input id="minChunks" type="range" min="0" max="200" value="0">
      </label>
      <div class="hint" id="rangeLabel"></div>
      <div class="hint">Density 用來找異常大章節；Spine 看分卷節奏；Index 可排序檢查具體章節。</div>
    </aside>

    <main>
      <section class="panel">
        <div class="panel-head">
          <h2 id="panelTitle">Chapter Density</h2>
          <span id="panelMeta"></span>
        </div>
        <div id="chart"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Chapter Index</h2>
          <span>click headers to sort</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th data-sort="volume">Volume</th>
                <th data-sort="chapter">Chapter</th>
                <th data-sort="chunks">Chunks</th>
                <th data-sort="edges">Edges</th>
              </tr>
            </thead>
            <tbody id="chapterRows"></tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const rawData = __DATA__;

const palette = ["#d7a84d", "#72b7bb", "#d47766", "#92b66b", "#9a8fcb", "#c8865d", "#70a0d8"];
const state = { mode: "bars", sortKey: "chunks", sortDir: -1 };

const flatAll = rawData.flatMap((book, bookIndex) =>
  book.chapters.map((chapter, index) => ({
    ...chapter,
    title: book.title,
    bookIndex,
    index,
    color: palette[bookIndex % palette.length],
  }))
);

const els = {
  metrics: document.getElementById("metrics"),
  titleFilter: document.getElementById("titleFilter"),
  volumeFilter: document.getElementById("volumeFilter"),
  searchInput: document.getElementById("searchInput"),
  minChunks: document.getElementById("minChunks"),
  rangeLabel: document.getElementById("rangeLabel"),
  chart: document.getElementById("chart"),
  rows: document.getElementById("chapterRows"),
  panelTitle: document.getElementById("panelTitle"),
  panelMeta: document.getElementById("panelMeta"),
  tooltip: document.getElementById("tooltip"),
};

function fmt(n) {
  return new Intl.NumberFormat("en-US").format(n);
}

function stripVolume(chapter) {
  return chapter.replace(/^第\S+卷\s*/, "");
}

function initControls() {
  els.titleFilter.innerHTML = `<option value="all">全部書籍</option>` +
    rawData.map(d => `<option value="${escapeHtml(d.title)}">${escapeHtml(d.title)}</option>`).join("");
  updateVolumeFilter();

  document.querySelectorAll("#modeButtons button").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll("#modeButtons button").forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      state.mode = button.dataset.mode;
      render();
    });
  });

  [els.titleFilter, els.volumeFilter, els.searchInput, els.minChunks].forEach(el => {
    el.addEventListener("input", () => {
      if (el === els.titleFilter) updateVolumeFilter();
      render();
    });
  });

  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir *= -1;
      else {
        state.sortKey = key;
        state.sortDir = key === "chapter" || key === "volume" ? 1 : -1;
      }
      renderTable(filteredData());
    });
  });
}

function updateVolumeFilter() {
  const selectedTitle = els.titleFilter.value || "all";
  const volumes = [...new Set(flatAll
    .filter(d => selectedTitle === "all" || d.title === selectedTitle)
    .map(d => d.volume))];
  els.volumeFilter.innerHTML = `<option value="all">全部分卷</option>` +
    volumes.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
}

function filteredData() {
  const title = els.titleFilter.value || "all";
  const volume = els.volumeFilter.value || "all";
  const q = els.searchInput.value.trim().toLowerCase();
  const min = Number(els.minChunks.value);
  return flatAll.filter(d =>
    (title === "all" || d.title === title) &&
    (volume === "all" || d.volume === volume) &&
    d.chunks >= min &&
    (!q || `${d.title} ${d.volume} ${d.chapter}`.toLowerCase().includes(q))
  );
}

function renderMetrics(data) {
  const totalChunks = d3.sum(data, d => d.chunks);
  const totalEdges = d3.sum(data, d => d.edges);
  const volumes = new Set(data.map(d => d.volume)).size;
  const maxChapter = data.length ? d3.max(data, d => d.chunks) : 0;
  els.metrics.innerHTML = [
    ["Chapters", data.length],
    ["Volumes", volumes],
    ["Chunks", totalChunks],
    ["Edges", totalEdges],
  ].map(([label, value]) => `<div class="metric"><b>${fmt(value)}</b><span>${label}</span></div>`).join("");
  els.rangeLabel.textContent = `目前門檻：至少 ${minLabel()} chunks；最大章節 ${fmt(maxChapter)} chunks。`;
}

function minLabel() {
  return fmt(Number(els.minChunks.value));
}

function render() {
  const data = filteredData();
  renderMetrics(data);
  renderTable(data);
  if (state.mode === "bars") renderBars(data);
  if (state.mode === "timeline") renderTimeline(data);
  if (state.mode === "table") renderTableOnly(data);
}

function setupSvg(width, height, margin) {
  els.chart.innerHTML = "";
  return d3.select(els.chart)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
}

function renderBars(data) {
  els.panelTitle.textContent = "Chapter Density";
  els.panelMeta.textContent = `${fmt(data.length)} chapters`;
  if (!data.length) return emptyChart();

  const sorted = [...data].sort((a, b) => b.chunks - a.chunks).slice(0, 120);
  const margin = { top: 22, right: 22, bottom: 130, left: 66 };
  const width = Math.max(920, sorted.length * 13);
  const height = 520;
  const svg = setupSvg(width, height, margin);
  const x = d3.scaleBand().domain(sorted.map((_, i) => i)).range([0, width]).padding(0.2);
  const y = d3.scaleLinear().domain([0, d3.max(sorted, d => d.chunks) * 1.08]).nice().range([height, 0]);

  svg.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(7).tickSize(-width).tickFormat(""));
  svg.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(7));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).tickValues([]));

  svg.selectAll("rect")
    .data(sorted)
    .join("rect")
    .attr("x", (_, i) => x(i))
    .attr("y", d => y(d.chunks))
    .attr("width", x.bandwidth())
    .attr("height", d => height - y(d.chunks))
    .attr("fill", d => d.color)
    .attr("opacity", 0.88)
    .on("mousemove", (event, d) => showTip(event, tipHtml(d)))
    .on("mouseleave", hideTip);

  svg.selectAll(".label")
    .data(sorted.filter((_, i) => i % Math.ceil(sorted.length / 42) === 0))
    .join("text")
    .attr("x", d => x(sorted.indexOf(d)) + x.bandwidth() / 2)
    .attr("y", height + 12)
    .attr("transform", d => `rotate(-58, ${x(sorted.indexOf(d)) + x.bandwidth() / 2}, ${height + 12})`)
    .attr("text-anchor", "end")
    .attr("fill", "#aaa18e")
    .attr("font-size", 11)
    .text(d => stripVolume(d.chapter).slice(0, 22));
}

function renderTimeline(data) {
  els.panelTitle.textContent = "Volume Spine";
  els.panelMeta.textContent = "width = chunk volume";
  if (!data.length) return emptyChart();

  const grouped = d3.groups(data, d => d.volume).map(([volume, chapters]) => ({
    volume,
    chunks: d3.sum(chapters, d => d.chunks),
    chapters,
  }));
  const margin = { top: 24, right: 30, bottom: 34, left: 190 };
  const width = Math.max(900, els.chart.clientWidth - margin.left - margin.right - 2);
  const rowH = 46;
  const height = Math.max(420, grouped.length * rowH);
  const svg = setupSvg(width, height, margin);
  const x = d3.scaleLinear().domain([0, d3.max(grouped, d => d.chunks)]).nice().range([0, width]);

  svg.append("g")
    .attr("class", "grid")
    .call(d3.axisTop(x).ticks(7).tickSize(-height).tickFormat(""));

  grouped.forEach((group, row) => {
    const y = row * rowH + 10;
    svg.append("text")
      .attr("x", -14)
      .attr("y", y + 15)
      .attr("text-anchor", "end")
      .attr("fill", "#d7a84d")
      .attr("font-size", 13)
      .text(group.volume);

    let cursor = 0;
    group.chapters.forEach(ch => {
      const w = Math.max(2, x(ch.chunks));
      svg.append("rect")
        .attr("x", cursor)
        .attr("y", y)
        .attr("width", w)
        .attr("height", 26)
        .attr("fill", ch.color)
        .attr("opacity", 0.82)
        .on("mousemove", event => showTip(event, tipHtml(ch)))
        .on("mouseleave", hideTip);
      cursor += w + 2;
    });

    svg.append("text")
      .attr("x", Math.min(cursor + 8, width - 8))
      .attr("y", y + 17)
      .attr("fill", "#aaa18e")
      .attr("font-family", "Consolas, Courier New, monospace")
      .attr("font-size", 12)
      .text(`${fmt(group.chunks)} chunks`);
  });
}

function renderTableOnly(data) {
  els.panelTitle.textContent = "Index Mode";
  els.panelMeta.textContent = "use the table below";
  els.chart.innerHTML = `<div class="empty">主視圖已切換到章節索引。使用下方表格排序與搜尋。</div>`;
}

function renderTable(data) {
  const sorted = [...data].sort((a, b) => {
    const av = a[state.sortKey];
    const bv = b[state.sortKey];
    if (typeof av === "number") return (av - bv) * state.sortDir;
    return String(av).localeCompare(String(bv), "zh-Hant") * state.sortDir;
  });
  els.rows.innerHTML = sorted.map(d => `
    <tr>
      <td>${escapeHtml(d.volume)}</td>
      <td>${escapeHtml(d.chapter)}</td>
      <td class="num">${fmt(d.chunks)}</td>
      <td class="num">${fmt(d.edges)}</td>
    </tr>
  `).join("");
}

function emptyChart() {
  els.chart.innerHTML = `<div class="empty">沒有符合篩選條件的章節。</div>`;
}

function tipHtml(d) {
  return `<b>${escapeHtml(d.chapter)}</b><br>${escapeHtml(d.title)}<br>Volume: ${escapeHtml(d.volume)}<br>Chunks: <b>${fmt(d.chunks)}</b><br>Edges: ${fmt(d.edges)}`;
}

function showTip(event, html) {
  els.tooltip.innerHTML = html;
  els.tooltip.style.opacity = 1;
  els.tooltip.style.left = `${event.clientX + 16}px`;
  els.tooltip.style.top = `${event.clientY - 8}px`;
}

function hideTip() {
  els.tooltip.style.opacity = 0;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("resize", () => {
  if (state.mode === "timeline") render();
});

initControls();
render();
</script>
</body>
</html>
"""

OUT_PATH.write_text(
    html.replace("__DATA__", json.dumps(data, ensure_ascii=True)),
    encoding="utf-8",
)

print(f"Done: {OUT_PATH}")
