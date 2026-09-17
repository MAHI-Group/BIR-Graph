"""Run with `python tests/test_birgraph.py` or `pytest tests`."""

import itertools
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_mice_protein  # noqa: E402
import run_birgraph  # noqa: E402
import synthetic  # noqa: E402
from birgraph import clusters, mining, network, stepminer  # noqa: E402


def stepmine_reference(x):
    """Quadratic-time split search as in BIRDNet's bir.stepmine."""
    xs = np.sort(x)
    n = len(xs)
    best_ssr, best_thr = np.inf, xs[n // 2]
    for k in range(1, n - 1):
        lo, hi = xs[: k + 1], xs[k + 1:]
        ssr = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        if ssr < best_ssr:
            best_ssr, best_thr = ssr, (xs[k] + xs[k + 1]) / 2.0
    return best_thr


def test_stepminer_matches_reference():
    rng = np.random.default_rng(1)
    for n in (5, 17, 60, 200):
        x = np.concatenate([rng.normal(4, 1, n // 2), rng.normal(8, 1.5, n - n // 2)])
        assert np.isclose(stepminer.stepminer_threshold(x)[0], stepmine_reference(x))


def planted_states(relation, n=400, seed=0):
    rng = np.random.default_rng(seed)
    a, u = rng.random(n) < 0.5, rng.random(n) < 0.5
    b = {"low=>low": a & u, "high=>high": a | u, "high=>low": ~a & u, "low=>high": ~a | u,
         "equivalent": a.copy(), "opposite": ~a}[relation]
    return a, b


def test_planted_relations_recovered():
    rng = np.random.default_rng(5)
    for relation in mining.RELATIONS:
        a, b = planted_states(relation)
        X = 3.0 + 4.0 * np.column_stack([a, b]) + rng.normal(0.0, 0.3, (a.size, 2))
        _, hi, lo = stepminer.fit_thresholds(pd.DataFrame(X, columns=["A", "B"]))
        for method in ("sahoo", "binomial"):
            rel, _ = mining.mine_relations(hi, lo, ["A", "B"], method=method)
            assert list(rel["relation"]) == [relation], (relation, method, list(rel["relation"]))


def test_independent_genes_give_no_relations():
    rng = np.random.default_rng(3)
    X = 3.0 + 4.0 * (rng.random((300, 30)) < 0.5) + rng.normal(0.0, 0.3, (300, 30))
    expr = pd.DataFrame(X, columns=[f"g{k}" for k in range(30)])
    _, hi, lo = stepminer.fit_thresholds(expr)
    rel, _ = mining.mine_relations(hi, lo, list(expr.columns))
    assert len(rel) == 0


def test_synthetic_structure_recovered():
    table, _ = synthetic.simulate()
    expr = table.drop(columns=["SampleID", "Group"])
    genes, hi, lo = stepminer.fit_thresholds(expr)
    ok = genes["passes_dynamic_range"].to_numpy()
    assert set(genes.loc[~ok, "gene"]) == {"L01", "L02"}
    names = genes.loc[ok, "gene"].tolist()
    rel, counts = mining.mine_relations(hi[:, ok], lo[:, ok], names)
    fdr = mining.permutation_fdr(hi[:, ok], lo[:, ok], counts, n_perm=3)
    assert (fdr["fdr"].fillna(0) < 0.05).all()
    labels, ctab = clusters.assign_clusters(names, rel)
    crel = clusters.cluster_relations(rel, labels, ctab)
    members = dict(zip(ctab["cluster"], ctab["members"]))
    module = {}
    for c in set(crel["source"]) | set(crel["target"]):
        letters = {g[0] for g in members[c].split(", ")}
        assert len(letters) == 1, (c, members[c])
        module[c] = letters.pop()
    found = {(module[s], k, module[t]) for s, t, k in zip(crel["source"], crel["target"], crel["kind"])}
    assert found == {("B", "HH", "A"), ("C", "OP", "D"), ("E", "HL", "F"), ("G", "LH", "H")}, found


def test_layout_has_no_overlaps():
    rng = np.random.default_rng(0)
    graph = nx.gnm_random_graph(80, 160, seed=2, directed=True)
    groups = {v: int(rng.integers(0, 8)) for v in graph}
    radius = {v: float(rng.uniform(5.0, 14.0)) for v in graph}
    pos = network.layout(graph, radius, groups)
    for u, v in itertools.combinations(pos, 2):
        assert np.linalg.norm(pos[u] - pos[v]) >= radius[u] + radius[v] - 1e-6


def test_command_line_smoke():
    out = Path(tempfile.mkdtemp())
    table, truth = synthetic.simulate()
    xlsx = out / "synthetic.xlsx"
    synthetic.write_workbook(table, truth, xlsx)
    run_birgraph.run(run_birgraph.parse_args(["--input", str(xlsx), "--group-col", "Group",
                                              "--group-order", "Normal,Tumour", "--n-perm", "2",
                                              "--out", str(out / "results")]))
    for name in ("gene_network.png", "gene_network.html", "cluster_network.pdf", "relations.csv",
                 "birgraph_report.xlsx"):
        assert (out / "results" / name).exists(), name


def test_margin_sd_is_scale_free():
    table, _ = synthetic.simulate()
    expr = table.drop(columns=["SampleID", "Group"])
    rng = np.random.default_rng(4)
    scaled = expr * rng.uniform(0.1, 10.0, expr.shape[1]) + rng.uniform(-5.0, 5.0, expr.shape[1])
    found = []
    for data in (expr, scaled):
        genes, hi, lo = stepminer.fit_thresholds(data, margin_sd=0.3)
        ok = genes["passes_dynamic_range"].to_numpy()
        rel, _ = mining.mine_relations(hi[:, ok], lo[:, ok], genes.loc[ok, "gene"].tolist())
        found.append(set(zip(rel["gene_a"], rel["gene_b"], rel["relation"])))
    assert found[0] == found[1] and len(found[0]) > 0


def mock_mice_table(n_mice=60, n_rep=15, seed=0):
    """Table with the layout of the UCI file: MouseID <mouse>_<replicate>, protein
    columns ending in _N, Genotype, Treatment, Behavior and class."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mice):
        trisomic = m % 2 == 1
        level = 0.8 * trisomic + rng.normal(0.0, 0.1)
        treatment = "Memantine" if m % 4 < 2 else "Saline"
        behaviour = "C/S" if m % 3 else "S/C"
        label = f"{'t' if trisomic else 'c'}-{'CS' if behaviour == 'C/S' else 'SC'}-{treatment[0].lower()}"
        for r in range(1, n_rep + 1):
            logs = np.array([0.2 + level, 0.5 + level, -0.3 + level, 1.0 - level, 0.0, 0.3]) + rng.normal(0.0, 0.15, 6)
            rows.append({"MouseID": f"{300 + m}_{r}", **{f"P{k}_N": v for k, v in enumerate(np.exp(logs), start=1)},
                         "Genotype": "Ts65Dn" if trisomic else "Control", "Treatment": treatment,
                         "Behavior": behaviour, "class": label})
    raw = pd.DataFrame(rows)
    raw.loc[raw.sample(frac=0.05, random_state=1).index, "P2_N"] = np.nan
    raw.loc[raw["MouseID"].str.startswith("301_"), "P5_N"] = np.nan
    return raw


def test_prepare_mice_protein_and_run():
    raw = mock_mice_table()
    replicates, mice = prepare_mice_protein.prepare(raw)
    assert replicates.shape == (900, 11) and mice.shape == (60, 11)
    assert list(mice.columns[:6]) == ["SampleID", "Genotype", "Treatment", "Behavior", "class", "P1"]
    first = raw[raw["MouseID"].str.startswith("300_")]
    assert np.isclose(mice.loc[mice["SampleID"] == "300", "P1"].item(), np.log2(first["P1_N"]).mean())
    assert mice.loc[mice["SampleID"] == "301", "P5"].isna().all()
    out = Path(tempfile.mkdtemp())
    prepare_mice_protein.write(mice, out / "mice.xlsx", ["test"])
    run_birgraph.run(run_birgraph.parse_args(["--input", str(out / "mice.xlsx"), "--group-col", "Genotype",
                                              "--group-order", "Control,Ts65Dn", "--margin-sd", "0.25",
                                              "--n-perm", "2", "--out", str(out / "results")]))
    rel = pd.read_csv(out / "results" / "relations.csv")
    assert {"P1", "P2", "P3", "P4"} <= set(rel["gene_a"]) | set(rel["gene_b"])
    assert (out / "results" / "relation_examples.png").exists()


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"passed {name}")
