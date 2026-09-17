import html
import json
import math
from collections import Counter

import numpy as np

from .network import KIND_COLOUR, KIND_DIRECTED, KIND_LABEL, KIND_STYLE, KIND_TEMPLATE, LABEL_FONT

DASH = {"solid": "", "dashed": "6 4", "dotted": "2 3"}
MAX_LISTED = 200

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ink: #1d2733; --muted: #5b6674; --panel: #f1f3f6; --rule: #d6dbe2; }
* { box-sizing: border-box; }
body { margin: 0; height: 100vh; display: grid; grid-template-columns: 300px 1fr;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: var(--ink); }
aside { background: var(--panel); border-right: 1px solid var(--rule); padding: 20px 18px; overflow: auto; }
h1 { font-size: 17px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
.meta, .hint { color: var(--muted); }
.meta { margin: 0 0 18px; }
.hint { font-size: 12px; }
fieldset { border: 0; margin: 0 0 18px; padding: 0; }
legend { font-weight: 600; margin-bottom: 6px; padding: 0; }
input[type=search] { width: 100%; padding: 6px 8px; font: inherit; border: 1px solid var(--rule); border-radius: 4px; }
label.kind { display: grid; grid-template-columns: 18px 30px 1fr; align-items: center; gap: 6px; padding: 3px 0; cursor: pointer; }
.swatch { height: 0; border-top-width: 3px; }
#selected { border-top: 1px solid var(--rule); padding-top: 14px; margin-top: 4px; }
.info { font-size: 13px; margin-bottom: 8px; }
.rels { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.rels li { border-left: 3px solid; padding: 1px 0 1px 7px; margin-bottom: 3px; }
main { position: relative; overflow: hidden; background: #fff; }
svg { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
svg text { font-size: __FONT__px; fill: var(--ink); pointer-events: none; text-anchor: middle; dominant-baseline: central;
  paint-order: stroke; stroke: #fff; stroke-width: 2px; stroke-linejoin: round; }
.faded { opacity: 0.1; }
.tip { position: absolute; display: none; pointer-events: none; max-width: 280px; background: #fff;
  border: 1px solid var(--rule); border-radius: 4px; padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 10px rgba(29, 39, 51, 0.12); }
.tip strong { display: block; font-size: 13px; margin-bottom: 2px; }
@media (max-width: 720px) { body { grid-template-columns: 1fr; grid-template-rows: auto 70vh; height: auto; } }
</style>
</head>
<body>
<aside>
  <h1>__TITLE__</h1>
  <p class="meta" id="meta"></p>
  <fieldset>
    <legend><label for="search">Find a node</label></legend>
    <input type="search" id="search" list="names" placeholder="Type a name and press Enter">
    <datalist id="names"></datalist>
  </fieldset>
  <fieldset id="kinds"><legend>Relations shown</legend></fieldset>
  <p class="hint">Scroll to zoom and drag to pan. Click a node to focus on its neighbourhood and list its relations; click empty space or press Escape to reset.</p>
  <section id="selected" hidden>
    <h2 id="sel-name"></h2>
    <div class="info" id="sel-info"></div>
    <ul class="rels" id="sel-rels"></ul>
  </section>
</aside>
<main>
  <svg id="net" role="img" aria-label="Boolean implication network">
    <defs id="defs"></defs>
    <g id="view"><g id="edges"></g><g id="nodes"></g></g>
  </svg>
  <div class="tip" id="tip"></div>
</main>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("net");
const view = document.getElementById("view");
const tip = document.getElementById("tip");
const kindOrder = Object.keys(DATA.kinds);
const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const adj = new Map(DATA.nodes.map(n => [n.id, new Set()]));
DATA.edges.forEach(e => { adj.get(e.s).add(e.t); adj.get(e.t).add(e.s); });
document.getElementById("meta").textContent = `${DATA.nodes.length} nodes, ${DATA.edges.length} relations`;

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const statement = e => DATA.kinds[e.k].template.replace("{s}", () => e.s).replace("{t}", () => e.t);
function make(tag, attrs, parent) {
  const x = document.createElementNS(NS, tag);
  for (const a in attrs) x.setAttribute(a, attrs[a]);
  if (parent) parent.appendChild(x);
  return x;
}

for (const [k, s] of Object.entries(DATA.kinds)) {
  if (!s.directed) continue;
  const m = make("marker", {id: "arrow-" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto"}, document.getElementById("defs"));
  make("path", {d: "M0,0 L10,5 L0,10 z", fill: s.colour}, m);
}

const edgeLayer = document.getElementById("edges");
const edgeEls = DATA.edges.map(e => {
  const a = byId.get(e.s), b = byId.get(e.t), s = DATA.kinds[e.k];
  const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
  const cut = s.directed ? b.r + 1 : 0;
  const line = make("line", {x1: a.x, y1: a.y, x2: b.x - dx / len * cut, y2: b.y - dy / len * cut,
    stroke: s.colour, "stroke-width": 1.4, "stroke-opacity": 0.8}, edgeLayer);
  if (s.dash) line.setAttribute("stroke-dasharray", s.dash);
  if (s.directed) line.setAttribute("marker-end", `url(#arrow-${e.k})`);
  line.dataset.k = e.k;
  line.dataset.s = e.s;
  line.dataset.t = e.t;
  return line;
});

const nodeLayer = document.getElementById("nodes");
const nodeEls = DATA.nodes.map(n => {
  const g = make("g", {transform: `translate(${n.x},${n.y})`}, nodeLayer);
  make("circle", {r: n.r, fill: n.c, stroke: "#27313c", "stroke-width": 0.6}, g);
  if (DATA.labels) make("text", {}, g).textContent = n.id;
  g.dataset.id = n.id;
  g.style.cursor = "pointer";
  g.addEventListener("pointermove", ev => showTip(ev, n));
  g.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  g.addEventListener("click", ev => { ev.stopPropagation(); setFocus(n.id); });
  return g;
});

function showTip(ev, n) {
  const rows = Object.entries(n.info).map(([k, v]) => `<div>${esc(k)}: ${esc(v)}</div>`).join("");
  tip.innerHTML = `<strong>${esc(n.id)}</strong>${rows}`;
  const box = svg.getBoundingClientRect();
  tip.style.left = Math.max(0, Math.min(ev.clientX - box.left + 14, box.width - 290)) + "px";
  tip.style.top = (ev.clientY - box.top + 14) + "px";
  tip.style.display = "block";
}

const hidden = new Set();
let focused = null;
function refresh() {
  const near = focused ? new Set([focused, ...adj.get(focused)]) : null;
  edgeEls.forEach(l => {
    l.style.display = hidden.has(l.dataset.k) ? "none" : "";
    l.classList.toggle("faded", !!near && l.dataset.s !== focused && l.dataset.t !== focused);
  });
  nodeEls.forEach(g => g.classList.toggle("faded", !!near && !near.has(g.dataset.id)));
}

function setFocus(id) {
  focused = id;
  refresh();
  const panel = document.getElementById("selected");
  if (!id) { panel.hidden = true; return; }
  const n = byId.get(id);
  document.getElementById("sel-name").textContent = n.id;
  document.getElementById("sel-info").replaceChildren(...Object.entries(n.info).map(([k, v]) => {
    const d = document.createElement("div");
    d.textContent = `${k}: ${v}`;
    return d;
  }));
  const rels = DATA.edges.filter(e => e.s === id || e.t === id)
    .sort((a, b) => kindOrder.indexOf(a.k) - kindOrder.indexOf(b.k));
  const items = rels.slice(0, __MAX__).map(e => {
    const li = document.createElement("li");
    li.textContent = statement(e);
    li.style.borderLeftColor = DATA.kinds[e.k].colour;
    return li;
  });
  if (rels.length > __MAX__) {
    const li = document.createElement("li");
    li.textContent = `and ${rels.length - __MAX__} more`;
    items.push(li);
  }
  document.getElementById("sel-rels").replaceChildren(...items);
  panel.hidden = false;
}

const kindBox = document.getElementById("kinds");
for (const [k, s] of Object.entries(DATA.kinds)) {
  const lab = document.createElement("label");
  lab.className = "kind";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = true;
  cb.addEventListener("change", () => { cb.checked ? hidden.delete(k) : hidden.add(k); refresh(); });
  const sw = document.createElement("span");
  sw.className = "swatch";
  sw.style.borderTopStyle = s.style;
  sw.style.borderTopColor = s.colour;
  lab.append(cb, sw, document.createTextNode(`${s.label} (${s.count})`));
  kindBox.appendChild(lab);
}

let k = 1, kFit = 1, tx = 0, ty = 0, drag = null, moved = false;
const kMax = () => Math.max(kFit * 40, 8);
const apply = () => view.setAttribute("transform", `translate(${tx},${ty}) scale(${k})`);
function fit() {
  const box = svg.getBoundingClientRect();
  kFit = Math.min(box.width / DATA.width, box.height / DATA.height) * 0.96;
  k = kFit;
  tx = (box.width - DATA.width * k) / 2;
  ty = (box.height - DATA.height * k) / 2;
  apply();
}
svg.addEventListener("wheel", ev => {
  ev.preventDefault();
  const box = svg.getBoundingClientRect(), mx = ev.clientX - box.left, my = ev.clientY - box.top;
  const nk = Math.min(kMax(), Math.max(kFit * 0.5, k * Math.exp(-ev.deltaY * 0.0015)));
  tx = mx - (mx - tx) * nk / k;
  ty = my - (my - ty) * nk / k;
  k = nk;
  apply();
}, {passive: false});
svg.addEventListener("pointerdown", ev => {
  drag = {x: ev.clientX - tx, y: ev.clientY - ty, x0: ev.clientX, y0: ev.clientY};
  moved = false;
  svg.style.cursor = "grabbing";
});
window.addEventListener("pointermove", ev => {
  if (!drag) return;
  tx = ev.clientX - drag.x;
  ty = ev.clientY - drag.y;
  moved = moved || Math.hypot(ev.clientX - drag.x0, ev.clientY - drag.y0) > 3;
  apply();
});
window.addEventListener("pointerup", () => { drag = null; svg.style.cursor = "grab"; });
svg.addEventListener("click", () => { if (!moved) setFocus(null); });
window.addEventListener("keydown", ev => { if (ev.key === "Escape") setFocus(null); });

const list = document.getElementById("names");
DATA.nodes.forEach(n => { const o = document.createElement("option"); o.value = n.id; list.appendChild(o); });
document.getElementById("search").addEventListener("change", ev => {
  const n = byId.get(ev.target.value.trim());
  if (!n) return;
  const box = svg.getBoundingClientRect();
  k = Math.min(kMax(), Math.max(k, 1.5));
  tx = box.width / 2 - n.x * k;
  ty = box.height / 2 - n.y * k;
  apply();
  setFocus(n.id);
});

window.addEventListener("resize", fit);
fit();
</script>
</body>
</html>
"""


def write_viewer(graph, pos, path, title, node_colour, node_size, node_info=None, margin=48.0):
    """Standalone HTML viewer for a network laid out in points (network.layout)."""
    nodes = [v for v in graph if v in pos]
    if not nodes:
        return
    P = np.array([pos[v] for v in nodes], dtype=float)
    R = np.array([math.sqrt(node_size[v]) / 2.0 for v in nodes])
    lo = (P - R[:, None]).min(axis=0) - margin
    hi = (P + R[:, None]).max(axis=0) + margin
    edges = [(u, v, d["kind"]) for u, v, d in graph.edges(data=True) if u in pos and v in pos]
    count = Counter(kind for _, _, kind in edges)
    data = {
        "width": round(float(hi[0] - lo[0]), 1),
        "height": round(float(hi[1] - lo[1]), 1),
        "labels": len(nodes) <= 400,
        "nodes": [{"id": str(v), "x": round(float(p[0] - lo[0]), 2), "y": round(float(hi[1] - p[1]), 2),
                   "r": round(float(r), 2), "c": node_colour[v], "info": (node_info or {}).get(v, {})}
                  for v, p, r in zip(nodes, P, R)],
        "edges": [{"s": str(u), "t": str(v), "k": kind} for u, v, kind in edges],
        "kinds": {kind: {"label": KIND_LABEL[kind], "colour": KIND_COLOUR[kind], "style": KIND_STYLE[kind],
                         "dash": DASH[KIND_STYLE[kind]], "directed": KIND_DIRECTED[kind],
                         "template": KIND_TEMPLATE[kind], "count": count[kind]}
                  for kind in KIND_LABEL if count[kind]},
    }
    payload = json.dumps(data).replace("</", "<\\/")
    page = (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__FONT__", f"{LABEL_FONT:g}")
            .replace("__MAX__", str(MAX_LISTED)).replace("__DATA__", payload))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
