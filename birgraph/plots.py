import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import Normalize, to_hex, to_rgb
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .mining import KINDS, SPARSE_QUADS
from .network import CHAR_WIDTH, KIND_COLOUR, KIND_DIRECTED, KIND_LABEL, KIND_SHORT, KIND_STYLE, LABEL_FONT

INK = "#27313C"
MUTED = "#8A939E"
BAND = "#E6E9ED"
SPARSE = "#F6CFCA"
SINGLETON = "#C9CED4"
GROUP_COLOURS = ["#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948"]
LABEL_LIMIT = 200
MAX_SIDE_IN = 13.0
MIN_SIZE_IN = (5.0, 3.5)
MAX_UPSCALE = 1.5
CLUSTER_FONT = 9.0
PAD_PT = 14.0


def _save(fig, stem, extra=()):
    for suffix, kw in ((".png", {"dpi": 200}), (".pdf", {})):
        fig.savefig(f"{stem}{suffix}", bbox_inches="tight", bbox_extra_artists=list(extra) or None, **kw)
    plt.close(fig)


def _is_dark(colour):
    rgb = np.array(to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722]) < 0.179


def _group_palette(n):
    if n <= len(GROUP_COLOURS):
        return GROUP_COLOURS[:n]
    return [to_hex(plt.get_cmap("tab20")(k % 20)) for k in range(n)]


def cluster_colours(table, min_size=2):
    palette = [to_hex(c) for name in ("Set3", "Pastel1", "Pastel2") for c in plt.get_cmap(name).colors]
    colours, k = {}, 0
    for c, s in zip(table["cluster"], table["size"]):
        if s >= min_size:
            colours[c] = palette[k % len(palette)]
            k += 1
        else:
            colours[c] = SINGLETON
    return colours


def values_to_colours(values, cmap="RdBu_r", limit=None):
    finite = [abs(v) for v in values.values() if np.isfinite(v)]
    limit = limit or (max(finite) if finite else 1.0) or 1.0
    norm = Normalize(-limit, limit)
    cm = plt.get_cmap(cmap)
    colours = {k: to_hex(cm(norm(v))) if np.isfinite(v) else SINGLETON for k, v in values.items()}
    return colours, norm, cm


def draw_network(graph, pos, stem, node_colour, node_size, radius, title="", labels=None,
                 node_legend=None, legend_title="Cluster", colourbar=None, font_size=LABEL_FONT):
    """Draw a network laid out by network.layout.

    One layout unit is one printed point, unless the network would exceed
    MAX_SIDE_IN or is smaller than MIN_SIZE_IN, in which case markers, lines
    and text are scaled together (enlarged at most MAX_UPSCALE times).
    labels: node -> text (None for no labels); node_legend: list of
    (label, colour); colourbar: (norm, cmap, label).
    """
    nodes = [v for v in graph if v in pos]
    if not nodes:
        return
    P = np.array([pos[v] for v in nodes], dtype=float)
    R = np.array([radius[v] for v in nodes])
    lo = (P - R[:, None]).min(axis=0) - PAD_PT
    hi = (P + R[:, None]).max(axis=0) + PAD_PT
    span = hi - lo
    fit = min(MAX_SIDE_IN * 72.0 / span[0], MAX_SIDE_IN * 72.0 / span[1])
    fill = min(MIN_SIZE_IN[0] * 72.0 / span[0], MIN_SIZE_IN[1] * 72.0 / span[1])
    scale = float(min(fit, max(1.0, min(fill, MAX_UPSCALE))))
    size_in = np.maximum(span * scale / 72.0, MIN_SIZE_IN)
    centre, half = (lo + hi) / 2.0, size_in * 72.0 / scale / 2.0

    fig = plt.figure(figsize=tuple(size_in))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    sizes = [node_size[v] * scale ** 2 for v in nodes]
    many = graph.number_of_edges() > 300
    present = []
    for kind in KINDS:
        edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == kind and u in pos and v in pos]
        if not edges:
            continue
        present.append(kind)
        arrow = dict(arrowstyle="-|>", arrowsize=max(5.0, 10.0 * scale),
                     connectionstyle="arc3,rad=0.05") if KIND_DIRECTED[kind] else {}
        nx.draw_networkx_edges(graph, pos, edgelist=edges, nodelist=nodes, node_size=sizes, ax=ax,
                               edge_color=KIND_COLOUR[kind], style=KIND_STYLE[kind],
                               width=max(0.5, 1.3 * math.sqrt(scale)), alpha=0.5 if many else 0.85,
                               arrows=KIND_DIRECTED[kind], **arrow)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=[node_colour[v] for v in nodes],
                           node_size=sizes, edgecolors=INK, linewidths=max(0.3, 0.6 * scale), ax=ax)
    font = font_size * scale
    if labels and font >= 3.5:
        halo = [patheffects.withStroke(linewidth=max(1.0, 2.2 * scale), foreground="white")]
        for v, s in zip(nodes, sizes):
            if v not in labels:
                continue
            text = str(labels[v])
            inside = CHAR_WIDTH * font * max(len(line) for line in text.split("\n")) <= math.sqrt(s)
            light = inside and _is_dark(node_colour[v])
            ax.text(pos[v][0], pos[v][1], text, fontsize=font, color="white" if light else INK, ha="center",
                    va="center", linespacing=1.0, path_effects=None if light else halo, zorder=5)

    ax.set_xlim(centre[0] - half[0], centre[0] + half[0])
    ax.set_ylim(centre[1] - half[1], centre[1] + half[1])
    relation_handles = [Line2D([0], [0], color=KIND_COLOUR[k], lw=2, linestyle=KIND_STYLE[k],
                               marker=">" if KIND_DIRECTED[k] else None, markersize=6, label=KIND_LABEL[k])
                        for k in present]
    node_handles = [Patch(facecolor=c, edgecolor=INK, linewidth=0.5, label=lab) for lab, c in node_legend or []]
    extra, top = [], 1.0
    renderer = fig.canvas.get_renderer()
    for handles, name in ((relation_handles, "Relation"), (node_handles, legend_title)):
        if not handles:
            continue
        leg = Legend(ax, handles, [h.get_label() for h in handles], title=name, loc="upper left",
                     bbox_to_anchor=(1.01, top), frameon=False, fontsize=8, title_fontsize=9, alignment="left")
        ax.add_artist(leg)
        extra.append(leg)
        box = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        top = box.y0 - 12.0 / (size_in[1] * 72.0)
    if colourbar:
        norm, cmap, label = colourbar
        h = size_in[1]
        cax = fig.add_axes((0.3, -0.42 / h, 0.4, 0.11 / h))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=8, color=INK)
        cax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    _save(fig, stem, extra)


def plot_relation_counts(fdr, stem):
    tab = fdr[fdr["kind"] != "all"]
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - 0.2, tab["observed"], 0.4, color=[KIND_COLOUR[k] for k in tab["kind"]], label="Observed")
    if tab["permuted_mean"].notna().any():
        ax.bar(x + 0.2, tab["permuted_mean"], 0.4, color=SINGLETON,
               yerr=tab["permuted_sd"].fillna(0), capsize=2, label="Permuted (mean)")
        top = float(max(tab["observed"].max(), 1))
        for xi, (o, f) in enumerate(zip(tab["observed"], tab["fdr"])):
            if o:
                ax.text(xi - 0.2, o + 0.02 * top, f"FDR {f:.2g}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [KIND_SHORT[k] for k in tab["kind"]], fontsize=8)
    ax.set_ylabel("Gene pairs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, stem)


def plot_relation_examples(expr, genes, relations, stem, groups=None, levels=None, clusters=None):
    """Strongest pair per relation kind, taken between different clusters where
    possible (except equivalence), with thresholds, intermediate bands and
    sparse quadrants. High => high pairs are drawn as source (x) => target (y)."""
    picks = []
    for kind in KINDS:
        sub = relations[relations["kind"] == kind]
        if clusters is not None and kind != "EQ":
            cross = sub[sub["gene_a"].map(clusters) != sub["gene_b"].map(clusters)]
            sub = cross if len(cross) else sub
        if len(sub):
            picks.append(sub.sort_values("S", ascending=False, kind="mergesort").iloc[0])
    if not picks:
        return
    thr = dict(zip(genes["gene"], genes["threshold"]))
    marg = dict(zip(genes["gene"], genes["margin"]))
    palette = _group_palette(len(levels) if levels else 0)
    cols = min(3, len(picks))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    coloured = groups is not None and bool(levels)
    for ax, rec in zip(axes.flat, picks):
        if rec["kind"] == "HH":
            a, b, quads = rec["source"], rec["target"], ["10"]
        else:
            a, b, quads = rec["gene_a"], rec["gene_b"], SPARSE_QUADS[rec["relation"]]
        x, y = expr[a].to_numpy(float), expr[b].to_numpy(float)
        ta, tb, ma, mb = thr[a], thr[b], marg[a], marg[b]
        x0, x1 = np.nanmin(x), np.nanmax(x)
        y0, y1 = np.nanmin(y), np.nanmax(y)
        px, py = 0.05 * (x1 - x0 or 1), 0.05 * (y1 - y0 or 1)
        x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
        for q in quads:
            xs = (x0, ta - ma) if q[0] == "0" else (ta + ma, x1)
            ys = (y0, tb - mb) if q[1] == "0" else (tb + mb, y1)
            ax.add_patch(Rectangle((xs[0], ys[0]), xs[1] - xs[0], ys[1] - ys[0], color=SPARSE, zorder=0))
        ax.axvspan(ta - ma, ta + ma, color=BAND, zorder=0)
        ax.axhspan(tb - mb, tb + mb, color=BAND, zorder=0)
        ax.axvline(ta, color=MUTED, lw=0.8)
        ax.axhline(tb, color=MUTED, lw=0.8)
        if coloured:
            for lv, colour in zip(levels, palette):
                m = (groups == lv).to_numpy()
                ax.scatter(x[m], y[m], s=9, color=colour, alpha=0.75, lw=0)
        else:
            ax.scatter(x, y, s=9, color=INK, alpha=0.6, lw=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{rec['canonical']}\nS = {rec['S']:.1f}, error rate = {rec['error_rate']:.3f}",
                     fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    spare = list(axes.flat)[len(picks):]
    for ax in spare:
        ax.set_axis_off()
    fig.tight_layout()
    handles = [Patch(color=BAND, label="Intermediate band (ignored)"), Patch(color=SPARSE, label="Sparse quadrant")]
    if coloured:
        handles += [Line2D([0], [0], marker="o", linestyle="", color=c, label=lv)
                    for lv, c in zip(levels, palette)]
    if spare:
        spare[0].legend(handles=handles, loc="center", frameon=False, fontsize=9)
        _save(fig, stem)
    else:
        leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 5),
                         frameon=False, fontsize=9)
        _save(fig, stem, [leg])


def plot_stepminer_examples(expr, genes, stem):
    """Best and weakest step fits among genes passing the dynamic-range filter, plus one failing gene."""
    ok = genes[genes["passes_dynamic_range"]].sort_values("step_r2", ascending=False, kind="mergesort")
    chosen = [(g, "best fit") for g in ok["gene"].iloc[:2]]
    if len(ok) > 2:
        chosen.append((ok["gene"].iloc[-1], "weakest fit"))
    failing = genes.loc[~genes["passes_dynamic_range"], "gene"]
    if len(failing):
        chosen.append((failing.iloc[0], "fails dynamic-range filter"))
    if not chosen:
        return
    thr = dict(zip(genes["gene"], genes["threshold"]))
    r2 = dict(zip(genes["gene"], genes["step_r2"]))
    marg = dict(zip(genes["gene"], genes["margin"]))
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.6 * len(chosen), 3.3), squeeze=False)
    for ax, (g, tag) in zip(axes.flat, chosen):
        v = np.sort(expr[g].to_numpy(float))
        v = v[np.isfinite(v)]
        t, margin = thr[g], marg[g]
        k = int(np.clip(np.searchsorted(v, t), 1, v.size - 1))
        ax.axhspan(t - margin, t + margin, color=BAND, zorder=0)
        ax.scatter(np.arange(v.size), v, s=6, color=INK, lw=0)
        ax.hlines([v[:k].mean(), v[k:].mean()], [0, k], [k, v.size], color=KIND_COLOUR["HH"], lw=2)
        ax.axhline(t, color=MUTED, lw=0.8)
        ax.set_title(f"{g} ({tag})\nthreshold {t:.2f}, step R2 {r2[g]:.2f}", fontsize=9, color=INK)
        ax.set_xlabel("Samples (sorted)")
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_ylabel("Expression")
    fig.tight_layout()
    _save(fig, stem)
