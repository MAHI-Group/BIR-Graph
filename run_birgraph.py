import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from birgraph import clusters, io, mining, network, plots, report, stepminer, viewer
from birgraph import groups as grp
from birgraph.network import KIND_LABEL


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Boolean implication networks of genes and gene clusters "
                                            "from a samples x genes table (xlsx, csv or tsv).")
    io_ = p.add_argument_group("input")
    io_.add_argument("--input", required=True)
    io_.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet)")
    io_.add_argument("--sample-col", default=None, help="sample ID column (default: first column)")
    io_.add_argument("--group-col", default=None, help="optional annotation column, e.g. Tumour/Normal")
    io_.add_argument("--group-order", default=None, help="comma-separated levels; differences are second minus first")
    io_.add_argument("--subset", default=None, help="analyse only samples with this group value")
    io_.add_argument("--genes-as-rows", action="store_true", help="table is genes x samples")
    io_.add_argument("--log2", action="store_true", help="apply log2(x + 1) to raw values")
    io_.add_argument("--gene-list", default=None, help="text file with one gene per line")
    io_.add_argument("--top-var", type=int, default=None, help="keep the N most variable genes")

    bir = p.add_argument_group("Boolean implication test")
    bir.add_argument("--method", choices=["sahoo", "binomial"], default="sahoo")
    bir.add_argument("--margin", type=float, default=0.5, help="intermediate band: threshold +/- margin")
    bir.add_argument("--margin-sd", type=float, default=None,
                     help="per-gene margin of this many robust standard deviations (replaces --margin)")
    bir.add_argument("--max-intermediate", type=float, default=2 / 3)
    bir.add_argument("--s-thr", type=float, default=3.0)
    bir.add_argument("--err-thr", type=float, default=0.1)
    bir.add_argument("--p-thr", type=float, default=1e-6, help="binomial method only")
    bir.add_argument("--sparse-frac", type=float, default=0.05, help="binomial method only")
    bir.add_argument("--min-frac", type=float, default=0.05, help="minimum low and high fraction per gene in a pair")
    bir.add_argument("--min-total", type=int, default=10, help="minimum non-intermediate samples per pair")
    bir.add_argument("--n-perm", type=int, default=20, help="permutations for FDR (0 to skip)")

    cl = p.add_argument_group("clusters")
    cl.add_argument("--clusters", choices=["components", "louvain", "map"], default="components")
    cl.add_argument("--cluster-map", default=None, help="table with columns gene, cluster (for --clusters map)")
    cl.add_argument("--cluster-sheet", default=None)
    cl.add_argument("--min-cluster-size", type=int, default=2)
    cl.add_argument("--min-support", type=float, default=0.5)

    p.add_argument("--max-plot-edges", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="birgraph_results")
    return p.parse_args(argv)


def fmt(x, digits=3):
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else f"{x:.{digits}g}"
    return str(x)


def plain(d):
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items() if pd.notna(v)}


def run(args):
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    expr, groups = io.load_expression(args.input, args.sheet, args.sample_col, args.group_col,
                                      args.genes_as_rows, args.log2)
    if args.subset is not None:
        if groups is None:
            raise SystemExit("--subset requires --group-col")
        keep = (groups == args.subset).to_numpy()
        expr, groups = expr.loc[keep], groups.loc[keep]
    n_input = expr.shape[1]
    gene_list = io.read_gene_list(args.gene_list) if args.gene_list else None
    expr = io.select_genes(expr, gene_list, args.top_var)
    print(f"Data: {expr.shape[0]} samples, {expr.shape[1]} genes selected of {n_input}")

    genes, hi, lo = stepminer.fit_thresholds(expr, args.margin, args.max_intermediate, args.margin_sd)
    ok = genes["passes_dynamic_range"].to_numpy()
    names = genes.loc[ok, "gene"].tolist()
    print(f"StepMiner: {len(names)} genes pass the dynamic-range filter")

    test = dict(method=args.method, s_thr=args.s_thr, err_thr=args.err_thr, p_thr=args.p_thr,
                sparse_frac=args.sparse_frac, min_frac=args.min_frac, min_total=args.min_total)
    relations, counts = mining.mine_relations(hi[:, ok], lo[:, ok], names,
                                              X=expr[names].to_numpy(float), **test)
    fdr = mining.permutation_fdr(hi[:, ok], lo[:, ok], counts, args.n_perm, args.seed, **test)
    fdr.insert(1, "label", fdr["kind"].map(KIND_LABEL).fillna("All relations"))
    print(f"Relations: {len(relations)} gene pairs")

    mapping = io.load_cluster_map(args.cluster_map, args.cluster_sheet) if args.clusters == "map" else None
    labels, ctab = clusters.assign_clusters(names, relations, args.clusters, mapping, args.seed)
    crel = clusters.cluster_relations(relations, labels, ctab, args.min_support, args.min_cluster_size)
    genes["cluster"] = genes["gene"].map(labels)
    big = ctab.loc[ctab["size"] >= args.min_cluster_size, "cluster"].tolist()

    gene_graph = network.build_graph(names, relations[["source", "target", "kind", "S", "error_rate"]])
    cluster_graph = network.build_graph(big, crel[["source", "target", "kind", "support", "n_pairs", "mean_S"]])
    genes = genes.merge(network.degree_table(gene_graph, "gene"), on="gene", how="left")
    ctab = ctab.merge(network.degree_table(cluster_graph, "cluster"), on="cluster", how="left")

    levels = None
    if groups is not None and groups.nunique() > 1:
        order = args.group_order.split(",") if args.group_order else None
        gstats, levels = grp.group_statistics(expr, hi, groups, order)
        genes = genes.merge(gstats, on="gene", how="left")
        if "delta" in gstats:
            ctab["mean_delta"] = ctab["cluster"].map(
                clusters.cluster_mean(dict(zip(genes["gene"], genes["delta"])), ctab))
    two_groups = levels is not None and len(levels) == 2
    contrast = f"{levels[1]} - {levels[0]}" if two_groups else ""

    colours = plots.cluster_colours(ctab, args.min_cluster_size)
    shown = network.strongest_subgraph(gene_graph, args.max_plot_edges)
    gsize = {g: float(np.clip(60 + 15 * shown.degree(g), 60, 500)) for g in shown}
    active = sum(1 for g in shown if shown.degree(g) > 0)
    gtext = {g: g for g in shown} if active <= plots.LABEL_LIMIT else None
    grad = network.node_radius(gsize, gtext)
    pos = network.layout(shown, grad, labels, args.seed)
    gcolour = {g: colours[labels[g]] for g in names}
    sized = [(c, s) for c, s in zip(ctab["cluster"], ctab["size"]) if s >= args.min_cluster_size]
    legend = [(f"{c} ({s} genes)", colours[c]) for c, s in sized[:12]]
    legend_title = "Cluster" if len(sized) <= 12 else f"Largest 12 of {len(sized)} clusters"
    trimmed = "" if shown is gene_graph else \
        f" (strongest {args.max_plot_edges} of {gene_graph.number_of_edges()} relations)"
    plots.draw_network(shown, pos, out / "gene_network", gcolour, gsize, grad,
                       title=f"Gene-level Boolean implication network{trimmed}", labels=gtext,
                       node_legend=legend, legend_title=legend_title)
    gt = genes.set_index("gene")
    if two_groups:
        dcol, norm, cm = plots.values_to_colours({g: gt.at[g, "delta"] for g in names})
        plots.draw_network(shown, pos, out / "gene_network_by_group", dcol, gsize, grad,
                           title=f"Gene-level network coloured by mean difference ({contrast})", labels=gtext,
                           colourbar=(norm, cm, f"Mean expression difference ({contrast})"))

    size_of = dict(zip(ctab["cluster"], ctab["size"]))
    ct = ctab.set_index("cluster")
    if cluster_graph.number_of_nodes():
        csize = {c: float(np.clip(250 + 90 * size_of[c], 300, 3000)) for c in cluster_graph}
        clabel = {c: f"{c}\n({size_of[c]})" for c in cluster_graph}
        crad = network.node_radius(csize, clabel, plots.CLUSTER_FONT)
        cpos = network.layout(cluster_graph, crad, seed=args.seed, keep_isolated=True)
        ccolour = {c: colours[c] for c in cluster_graph}
        plots.draw_network(cluster_graph, cpos, out / "cluster_network", ccolour, csize, crad,
                           title="Cluster-level Boolean implication network", labels=clabel,
                           font_size=plots.CLUSTER_FONT)
        if two_groups:
            dcol, norm, cm = plots.values_to_colours({c: ct.at[c, "mean_delta"] for c in cluster_graph})
            plots.draw_network(cluster_graph, cpos, out / "cluster_network_by_group", dcol, csize, crad,
                               title=f"Cluster-level network coloured by mean difference ({contrast})",
                               labels=clabel, colourbar=(norm, cm, f"Mean member difference ({contrast})"),
                               font_size=plots.CLUSTER_FONT)
        cinfo = {}
        for c in cluster_graph:
            members = ct.at[c, "members"].split(", ")
            info = {"genes": size_of[c], "members": ", ".join(members[:15]) + (" ..." if len(members) > 15 else ""),
                    "relations": int(ct.at[c, "degree"])}
            if two_groups:
                info[f"mean difference ({contrast})"] = f"{ct.at[c, 'mean_delta']:.2f}"
            cinfo[c] = info
        viewer.write_viewer(cluster_graph, cpos, out / "cluster_network.html",
                            f"Cluster network: {Path(args.input).name}", ccolour, csize, cinfo)

    ginfo = {}
    for g in shown:
        info = {"cluster": gt.at[g, "cluster"], "relations": int(gt.at[g, "degree"]),
                "threshold": f"{gt.at[g, 'threshold']:.2f}", "fraction high": f"{gt.at[g, 'frac_high']:.2f}"}
        if two_groups:
            info[f"mean difference ({contrast})"] = f"{gt.at[g, 'delta']:.2f}"
            info["q value"] = f"{gt.at[g, 'q_value']:.2g}"
        ginfo[g] = info
    viewer.write_viewer(shown, pos, out / "gene_network.html", f"Gene network: {Path(args.input).name}",
                        gcolour, gsize, ginfo)

    plots.plot_relation_counts(fdr, out / "relation_counts")
    plots.plot_relation_examples(expr, genes, relations, out / "relation_examples", groups, levels, labels)
    plots.plot_stepminer_examples(expr, genes, out / "stepminer_examples")

    node_cols = [k for k in ("cluster", "threshold", "frac_high", "delta", "q_value") if k in gt.columns]
    nx.set_node_attributes(gene_graph, {g: plain(gt.loc[g, node_cols].to_dict()) for g in names})
    nx.set_node_attributes(cluster_graph, {c: plain({"size": size_of[c]}) for c in cluster_graph})
    nx.write_graphml(gene_graph, out / "gene_network.graphml")
    nx.write_graphml(cluster_graph, out / "cluster_network.graphml")
    relations.to_csv(out / "relations.csv", index=False)
    crel.to_csv(out / "cluster_relations.csv", index=False)
    genes.to_csv(out / "genes.csv", index=False)

    gsum = network.network_summary(gene_graph)
    hubs = genes.dropna(subset=["degree"]).sort_values(["degree", "gene"], ascending=[False, True],
                                                        kind="mergesort").head(10)
    margin_rule = (f"margin {args.margin}" if args.margin_sd is None
                   else f"margin {args.margin_sd} robust SD per gene")
    rule = (f"S > {args.s_thr}, error rate < {args.err_thr}" if args.method == "sahoo"
            else f"binomial p < {args.p_thr}, exception fraction <= {args.sparse_frac}")
    rows = [
        ("Input file", str(args.input)),
        ("Samples", expr.shape[0]),
        ("Genes selected", f"{expr.shape[1]} of {n_input}"),
        ("Genes passing dynamic-range filter", len(names)),
        ("Test", f"{args.method}: {rule}; {margin_rule}"),
        ("Permutations for FDR", args.n_perm),
    ]
    if levels is not None:
        rows.append(("Groups", ", ".join(f"{lv} ({int((groups == lv).sum())})" for lv in levels)))
    rows += [(f"Relations, {r.label}", f"{r.observed} (FDR {fmt(r.fdr)})" if r.observed else "0")
             for r in fdr.itertuples()]
    rows += [(f"Gene network, {k.replace('_', ' ')}", fmt(v)) for k, v in gsum.items()]
    rows += [
        (f"Clusters with at least {args.min_cluster_size} genes", len(big)),
        (f"Cluster relations (support >= {args.min_support})", len(crel)),
        ("Top hubs (relations)", ", ".join(f"{g} ({int(d)})" for g, d in zip(hubs["gene"], hubs["degree"]))),
    ]
    summary = pd.DataFrame(rows, columns=["item", "value"])
    summary["value"] = summary["value"].astype(str)
    report.write_excel(out / "birgraph_report.xlsx", {
        "summary": summary, "genes": genes, "relations": relations, "fdr": fdr,
        "clusters": ctab, "cluster_relations": crel,
    })

    print()
    for item, value in rows:
        print(f"  {item}: {value}")
    if len(crel):
        print("\nCluster relations:")
        for s in crel["statement"]:
            print(f"  {s}")
    print(f"\nOutputs written to {out.resolve()} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    run(parse_args())
