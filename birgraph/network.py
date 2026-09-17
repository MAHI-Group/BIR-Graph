import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .mining import KINDS

KIND_LABEL = {
    "EQ": "Equivalent",
    "OP": "Opposite",
    "HH": "X high => Y high (arrow X to Y)",
    "HL": "high => low (rarely both high)",
    "LH": "low => high (rarely both low)",
}
KIND_SHORT = {"EQ": "Equivalent", "OP": "Opposite", "HH": "high => high", "HL": "high => low", "LH": "low => high"}
KIND_TEMPLATE = {"EQ": "{s} equivalent {t}", "OP": "{s} opposite {t}", "HH": "{s} high => {t} high",
                 "HL": "{s} high => {t} low", "LH": "{s} low => {t} high"}
KIND_COLOUR = {"EQ": "#009E73", "OP": "#D55E00", "HH": "#0072B2", "HL": "#CC79A7", "LH": "#E69F00"}
KIND_STYLE = {"EQ": "solid", "OP": "solid", "HH": "solid", "HL": "dashed", "LH": "dotted"}
KIND_DIRECTED = {"EQ": False, "OP": False, "HH": True, "HL": False, "LH": False}
LABEL_FONT = 7.0
CHAR_WIDTH = 0.62
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _clean(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_graph(nodes, edges, node_attrs=None):
    """Directed graph; only HH edges carry direction, other kinds are stored once."""
    graph = nx.DiGraph()
    for v in nodes:
        attrs = (node_attrs or {}).get(v, {})
        graph.add_node(v, **{k: _clean(x) for k, x in attrs.items() if pd.notna(x)})
    for rec in edges.to_dict("records"):
        u, v = rec.pop("source"), rec.pop("target")
        graph.add_edge(u, v, **{k: _clean(x) for k, x in rec.items() if pd.notna(x)})
    return graph


def degree_table(graph, key="gene"):
    rows = []
    for v in graph.nodes:
        row = {key: v, "degree": graph.degree(v), **{f"n_{k}": 0 for k in KINDS}, "hh_out": 0, "hh_in": 0}
        for _, _, d in graph.out_edges(v, data=True):
            row[f"n_{d['kind']}"] += 1
            row["hh_out"] += d["kind"] == "HH"
        for _, _, d in graph.in_edges(v, data=True):
            row[f"n_{d['kind']}"] += 1
            row["hh_in"] += d["kind"] == "HH"
        rows.append(row)
    columns = [key, "degree", *[f"n_{k}" for k in KINDS], "hh_out", "hh_in"]
    return pd.DataFrame(rows, columns=columns)


def network_summary(graph):
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    active = [v for v in graph if graph.degree(v) > 0]
    comps = [len(c) for c in nx.connected_components(graph.subgraph(active).to_undirected())]
    return {
        "nodes": n,
        "nodes_with_relations": len(active),
        "edges": e,
        "density": 2.0 * e / (n * (n - 1)) if n > 1 else 0.0,
        "connected_components": len(comps),
        "largest_component": max(comps, default=0),
    }


def strongest_subgraph(graph, max_edges):
    if graph.number_of_edges() <= max_edges:
        return graph
    ranked = sorted(graph.edges(data=True), key=lambda t: (-t[2].get("S", 0.0), t[0], t[1]))
    sub = nx.DiGraph()
    sub.add_nodes_from((v, graph.nodes[v]) for v in graph)
    sub.add_edges_from(ranked[:max_edges])
    return sub


def node_radius(node_size, text=None, font_size=LABEL_FONT):
    """Exclusion radius in points: the marker radius (node_size is a matplotlib
    marker area in points^2) or half the estimated label width, whichever is larger."""
    out = {}
    for v, s in node_size.items():
        r = math.sqrt(s) / 2.0
        if text is not None and v in text:
            longest = max(len(line) for line in str(text[v]).split("\n"))
            r = max(r, 0.5 * CHAR_WIDTH * font_size * longest)
        out[v] = r
    return out


def _spiral(n, spacing):
    """Vogel spiral of n points with minimum pairwise distance `spacing`, centred at the origin."""
    if n == 1:
        return np.zeros((1, 2))
    k = np.arange(n) + 0.5
    P = np.sqrt(k)[:, None] * np.column_stack((np.cos(k * GOLDEN), np.sin(k * GOLDEN)))
    d, _ = cKDTree(P).query(P, k=2)
    P = P * (spacing / d[:, 1].min())
    return P - P.mean(axis=0)


def _resolve_overlaps(P, R, gap, rng, iters=500):
    """Push discs of radius R apart until centres are at least R_i + R_j + gap apart."""
    n = len(P)
    if n < 2 or n > 3000:
        return P
    P = P + rng.normal(scale=1e-6 * (float(R.mean()) + 1.0), size=P.shape)
    need = R[:, None] + R[None, :] + gap
    for _ in range(iters):
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        overlap = need - dist
        if overlap.max() <= 0.5:
            break
        push = np.where(overlap > 0, overlap, 0.0) / np.maximum(dist, 1e-9)
        P = P + 0.45 * (diff * push[..., None]).sum(axis=1)
    return P


def _place_groups(sub, cc, grad, edge_len, gap, rng, seed):
    if len(cc) == 1:
        return np.zeros((1, 2))
    g = nx.Graph()
    g.add_nodes_from(cc)
    g.add_edges_from((u, v, {"length": grad[u] + grad[v] + edge_len}) for u, v in sub.edges())
    if len(cc) <= 300:
        raw = nx.kamada_kawai_layout(g, weight="length")
    else:
        raw = nx.spring_layout(g, seed=seed, iterations=200)
    at = {c: i for i, c in enumerate(cc)}
    P = np.array([raw[c] for c in cc], dtype=float)
    ratio = [d["length"] / max(float(np.linalg.norm(P[at[u]] - P[at[v]])), 1e-9)
             for u, v, d in g.edges(data=True)]
    P = P * float(np.median(ratio))
    return _resolve_overlaps(P, np.array([grad[c] for c in cc]), gap, rng)


def _landscape(P):
    P = P - P.mean(axis=0)
    if len(P) < 2:
        return P
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    return P @ vt.T


def _pack(blocks, gap):
    """Shelf-pack laid-out components in the given order into rows of similar width."""
    boxes = [((P - R[:, None]).min(axis=0), (P + R[:, None]).max(axis=0)) for P, R in blocks]
    area = sum(float(np.prod(hi - lo + gap)) for lo, hi in boxes)
    row_width = max(max(float(hi[0] - lo[0]) for lo, hi in boxes), 1.25 * math.sqrt(area))
    placed, x, y, row_h = [], 0.0, 0.0, 0.0
    for (P, _), (lo, hi) in zip(blocks, boxes):
        w, h = hi - lo
        if x > 0 and x + w > row_width:
            x, y, row_h = 0.0, y - row_h - gap, 0.0
        placed.append(P + np.array([x - lo[0], y - hi[1]]))
        x += w + gap
        row_h = max(row_h, float(h))
    return placed


def layout(graph, radius, groups=None, seed=0, keep_isolated=False):
    """Node positions in points, given exclusion radii in points (see node_radius).

    Members of a group (for example a cluster) sit on a compact spiral, largest
    nodes in the centre. Groups within a connected component are placed by
    Kamada-Kawai on the group graph with edge lengths set from group radii and
    then pushed apart until they do not overlap. Components are rotated to
    landscape and shelf-packed, largest first. Nodes do not overlap, so a figure
    can be drawn at a known scale from these positions.
    """
    nodes = [v for v in graph if keep_isolated or graph.degree(v) > 0]
    if not nodes:
        return {}
    groups = groups or {}
    rng = np.random.default_rng(seed)
    diam = float(np.median([2.0 * radius[v] for v in nodes]))
    node_gap, edge_len = max(3.0, 0.3 * diam), max(24.0, 2.0 * diam)
    group_gap, pack_gap = max(10.0, 0.8 * diam), max(20.0, 1.5 * diam)

    index = {v: k for k, v in enumerate(nodes)}
    key = {v: ("group", groups[v]) if v in groups else ("node", v) for v in nodes}
    members = {}
    for v in nodes:
        members.setdefault(key[v], []).append(v)
    offset, grad = {}, {}
    for c, vs in members.items():
        vs.sort(key=lambda v: (-radius[v], index[v]))
        P = _spiral(len(vs), 2.0 * max(radius[v] for v in vs) + node_gap)
        offset.update(zip(vs, P))
        grad[c] = max(float(np.hypot(*offset[v])) + radius[v] for v in vs)

    quotient = nx.Graph()
    quotient.add_nodes_from(members)
    quotient.add_edges_from((key[u], key[v]) for u, v in graph.edges()
                            if u in key and v in key and key[u] != key[v])
    first = {c: k for k, c in enumerate(members)}
    comps = sorted((sorted(cc, key=first.get) for cc in nx.connected_components(quotient)),
                   key=lambda cc: (-sum(len(members[c]) for c in cc), first[cc[0]]))
    blocks, order = [], []
    for cc in comps:
        centres = _place_groups(quotient.subgraph(cc), cc, grad, edge_len, group_gap, rng, seed)
        vs = [v for c in cc for v in members[c]]
        P = np.array([centres[i] + offset[v] for i, c in enumerate(cc) for v in members[c]])
        blocks.append((_landscape(P), np.array([radius[v] for v in vs])))
        order.append(vs)
    placed = _pack(blocks, pack_gap)
    return {v: p for vs, P in zip(order, placed) for v, p in zip(vs, P)}
