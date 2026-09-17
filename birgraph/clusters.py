import networkx as nx
import numpy as np
import pandas as pd


def _table(named):
    return pd.DataFrame({
        "cluster": [name for name, _ in named],
        "size": [len(m) for _, m in named],
        "members": [", ".join(m) for _, m in named],
    })


def assign_clusters(genes, relations, method="components", mapping=None, seed=0):
    """Group genes into clusters.

    components: connected components of the Boolean equivalence graph
    (Sahoo et al., 2008); louvain: Louvain communities of the same graph;
    map: a user-supplied gene -> cluster mapping (unmapped genes stay singletons).
    Returns (gene -> cluster dict, cluster table sorted by size).
    """
    genes = list(genes)
    order = {g: k for k, g in enumerate(genes)}
    if method == "map":
        if mapping is None:
            raise ValueError("method 'map' needs a gene -> cluster mapping")
        groups = {}
        for g in genes:
            groups.setdefault(mapping.get(g, g), []).append(g)
        named = sorted(groups.items(), key=lambda kv: (-len(kv[1]), order[kv[1][0]]))
    else:
        graph = nx.Graph()
        graph.add_nodes_from(genes)
        eq = relations[relations["kind"] == "EQ"]
        graph.add_edges_from(zip(eq["gene_a"], eq["gene_b"]))
        if method == "components":
            found = nx.connected_components(graph)
        elif method == "louvain":
            found = nx.community.louvain_communities(graph, seed=seed)
        else:
            raise ValueError(f"unknown cluster method '{method}'")
        groups = sorted((sorted(c, key=order.get) for c in found), key=lambda m: (-len(m), order[m[0]]))
        named = [(f"C{k + 1}", m) for k, m in enumerate(groups)]
    labels = {g: name for name, members in named for g in members}
    return labels, _table(named)


def cluster_relations(relations, labels, table, min_support=0.5, min_size=2):
    """Cluster-level relations from gene-level ones.

    For each pair of clusters, the most frequent gene-level relation kind
    (HH keeps its direction) is kept if it covers at least `min_support` of
    all member pairs. BoNE defines cluster edges from the dominant
    relationships between clusters (Sahoo et al., 2021); the support rule
    here is this package's operationalisation.
    """
    columns = ["source", "target", "kind", "statement", "n_pairs", "possible_pairs",
               "support", "purity", "mean_S"]
    size = dict(zip(table["cluster"], table["size"]))
    rank = {c: k for k, c in enumerate(table["cluster"])}
    df = relations[["source", "target", "kind", "S"]].copy()
    df["cs"] = df["source"].map(labels)
    df["ct"] = df["target"].map(labels)
    df = df[(df["cs"] != df["ct"])
            & (df["cs"].map(size) >= min_size)
            & (df["ct"].map(size) >= min_size)]
    if df.empty:
        return pd.DataFrame(columns=columns)

    forward = (df["cs"].map(rank) < df["ct"].map(rank)).to_numpy()
    df["c1"] = np.where(forward, df["cs"], df["ct"])
    df["c2"] = np.where(forward, df["ct"], df["cs"])
    df["key"] = np.where((df["kind"] == "HH").to_numpy() & ~forward, "HH_rev", df["kind"])
    grp = (df.groupby(["c1", "c2", "key"], sort=True)
             .agg(n_pairs=("S", "size"), mean_S=("S", "mean"))
             .reset_index())
    grp["purity"] = grp["n_pairs"] / grp.groupby(["c1", "c2"])["n_pairs"].transform("sum")
    grp["possible_pairs"] = grp["c1"].map(size) * grp["c2"].map(size)
    grp["support"] = grp["n_pairs"] / grp["possible_pairs"]
    grp = grp.sort_values(["c1", "c2", "n_pairs", "mean_S"],
                          ascending=[True, True, False, False], kind="mergesort")
    top = grp.drop_duplicates(["c1", "c2"], keep="first")
    top = top[top["support"] >= min_support]

    rev = (top["key"] == "HH_rev").to_numpy()
    out = pd.DataFrame({
        "source": np.where(rev, top["c2"], top["c1"]),
        "target": np.where(rev, top["c1"], top["c2"]),
        "kind": np.where(rev, "HH", top["key"]),
        "n_pairs": top["n_pairs"].to_numpy(),
        "possible_pairs": top["possible_pairs"].to_numpy(),
        "support": top["support"].to_numpy(),
        "purity": top["purity"].to_numpy(),
        "mean_S": top["mean_S"].to_numpy(),
    })
    form = {"EQ": "{s} equivalent {t}", "OP": "{s} opposite {t}", "HH": "{s} high => {t} high",
            "HL": "{s} high => {t} low", "LH": "{s} low => {t} high"}
    out["statement"] = [form[k].format(s=s, t=t)
                        for s, t, k in zip(out["source"], out["target"], out["kind"])]
    out = out.assign(_a=out["source"].map(rank), _b=out["target"].map(rank))
    out = out.sort_values(["_a", "_b"], kind="mergesort").drop(columns=["_a", "_b"])
    return out[columns].reset_index(drop=True)


def cluster_mean(values, table):
    """Mean of a per-gene quantity over the members of each cluster."""
    s = pd.Series(values)
    return {c: float(s.reindex(m.split(", ")).mean()) for c, m in zip(table["cluster"], table["members"])}
