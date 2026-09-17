import numpy as np
import pandas as pd
from scipy import stats

RELATIONS = ["low=>low", "low=>high", "high=>low", "high=>high", "equivalent", "opposite"]
QUADS = ("00", "01", "10", "11")
REL_OF_QUAD = {"01": "low=>low", "00": "low=>high", "11": "high=>low", "10": "high=>high"}
SPARSE_QUADS = {
    "low=>low": ["01"], "low=>high": ["00"], "high=>low": ["11"], "high=>high": ["10"],
    "equivalent": ["01", "10"], "opposite": ["00", "11"],
}
KINDS = ["EQ", "OP", "HH", "HL", "LH"]
KIND_OF = {
    "equivalent": "EQ", "opposite": "OP", "low=>low": "HH", "high=>high": "HH",
    "high=>low": "HL", "low=>high": "LH",
}


def quadrant_statistics(c):
    """Sparsity statistic S and error rate per quadrant (Sahoo et al., 2008).

    Quadrant keys are 'ab' with a, b in {0: low, 1: high} for genes A and B.
    """
    total = c["00"] + c["01"] + c["10"] + c["11"]
    a_lo, a_hi = c["00"] + c["01"], c["10"] + c["11"]
    b_lo, b_hi = c["00"] + c["10"], c["01"] + c["11"]
    margins = {"00": (a_lo, b_lo), "01": (a_lo, b_hi), "10": (a_hi, b_lo), "11": (a_hi, b_hi)}
    expected, S, err = {}, {}, {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for q, (na, nb) in margins.items():
            expected[q] = na * nb / total
            S[q] = (expected[q] - c[q]) / np.sqrt(expected[q])
            err[q] = 0.5 * (c[q] / na + c[q] / nb)
    return total, (a_lo, a_hi, b_lo, b_hi), expected, S, err


def _sparse_flags(c, method, s_thr, err_thr, p_thr, sparse_frac, min_frac, min_total):
    total, (a_lo, a_hi, b_lo, b_hi), expected, S, err = quadrant_statistics(c)
    valid = ((total >= min_total)
             & (np.minimum(a_lo, a_hi) >= min_frac * total)
             & (np.minimum(b_lo, b_hi) >= min_frac * total))
    sparse = {}
    for q in QUADS:
        if method == "sahoo":
            sparse[q] = valid & (S[q] > s_thr) & (err[q] < err_thr)
        elif method == "binomial":
            cand = valid & (c[q] <= sparse_frac * total) & (c[q] < expected[q])
            flag = np.zeros(cand.shape, dtype=bool)
            idx = np.nonzero(cand)
            if idx[0].size:
                p = stats.binom.cdf(c[q][idx], total[idx], expected[q][idx] / total[idx])
                flag[idx] = p < p_thr
            sparse[q] = flag
        else:
            raise ValueError(f"unknown method '{method}'")
    return sparse, total, expected, S, err


def _standardise(X):
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def mine_relations(hi, lo, names=None, method="sahoo", s_thr=3.0, err_thr=0.1,
                   p_thr=1e-6, sparse_frac=0.05, min_frac=0.05, min_total=10,
                   X=None, counts_only=False, max_block_cells=4_000_000):
    """Boolean implication relationships for all gene pairs (i < j).

    hi, lo: boolean (n_samples, n_genes) arrays; a sample is ignored for a
    pair if it is intermediate for either gene.
    method 'sahoo': S > s_thr and error rate < err_thr (Sahoo et al., 2008).
    method 'binomial': one-sided binomial test with an exception-fraction cap,
    as in BIRDNet's bir.test_bir.
    Returns a relation table and a count dict, or only the count dict.
    """
    H = np.ascontiguousarray(hi, dtype=np.float32)
    L = np.ascontiguousarray(lo, dtype=np.float32)
    n, G = H.shape
    block = int(max(1, min(G, max_block_cells // max(G, 1))))
    Z = _standardise(X) if (X is not None and not counts_only) else None
    counts = dict.fromkeys(RELATIONS, 0)
    parts = []
    cols = np.arange(G)

    for s0 in range(0, G, block):
        s1 = min(G, s0 + block)
        Hb, Lb = H[:, s0:s1], L[:, s0:s1]
        c = {"00": Lb.T @ L, "01": Lb.T @ H, "10": Hb.T @ L, "11": Hb.T @ H}
        c = {q: v.astype(np.float64) for q, v in c.items()}
        sparse, total, expected, S, err = _sparse_flags(
            c, method, s_thr, err_thr, p_thr, sparse_frac, min_frac, min_total)
        upper = cols[None, :] > np.arange(s0, s1)[:, None]
        s = {q: sparse[q] & upper for q in QUADS}
        eq = s["01"] & s["10"]
        op = s["00"] & s["11"] & ~eq
        ii, jj = np.nonzero(s["00"] | s["01"] | s["10"] | s["11"])
        if ii.size == 0:
            continue

        Sq = np.stack([np.where(s[q][ii, jj], S[q][ii, jj], -np.inf) for q in QUADS])
        best = np.argmax(Sq, axis=0)
        rel = np.array([REL_OF_QUAD[q] for q in QUADS], dtype=object)[best]
        eqm, opm = eq[ii, jj], op[ii, jj]
        rel[eqm] = "equivalent"
        rel[opm] = "opposite"
        if counts_only:
            for r, k in zip(*np.unique(rel.astype(str), return_counts=True)):
                counts[r] += int(k)
            continue

        tot = total[ii, jj]
        Se = {q: S[q][ii, jj] for q in QUADS}
        Ee = {q: err[q][ii, jj] for q in QUADS}
        with np.errstate(divide="ignore", invalid="ignore"):
            Pe = {q: stats.binom.cdf(c[q][ii, jj], tot, expected[q][ii, jj] / tot) for q in QUADS}
        S_e, E_e, P_e = (np.choose(best, [d[q] for q in QUADS]) for d in (Se, Ee, Pe))
        for mask, (q1, q2) in ((eqm, ("01", "10")), (opm, ("00", "11"))):
            S_e[mask] = np.minimum(Se[q1], Se[q2])[mask]
            E_e[mask] = np.maximum(Ee[q1], Ee[q2])[mask]
            P_e[mask] = np.maximum(Pe[q1], Pe[q2])[mask]

        part = pd.DataFrame({
            "i": ii + s0, "j": jj, "relation": rel.astype(str),
            "S": S_e, "error_rate": E_e, "binom_p": P_e, "n": tot.astype(int),
            "n_lowlow": c["00"][ii, jj].astype(int), "n_lowhigh": c["01"][ii, jj].astype(int),
            "n_highlow": c["10"][ii, jj].astype(int), "n_highhigh": c["11"][ii, jj].astype(int),
        })
        if Z is not None:
            part["pearson_r"] = ((Z[:, s0:s1].T @ Z)[ii, jj] / max(n - 1, 1)).astype(float)
        parts.append(part)

    if counts_only:
        return counts
    if not parts:
        empty = pd.DataFrame(columns=["gene_a", "gene_b", "relation", "statement", "kind",
                                      "source", "target", "canonical", "S", "error_rate", "binom_p", "n"])
        return empty, counts

    df = pd.concat(parts, ignore_index=True)
    names = np.asarray(names if names is not None else [f"g{k}" for k in range(G)], dtype=object)
    df = _annotate(df, names)
    order = {r: k for k, r in enumerate(RELATIONS)}
    df["_r"] = df["relation"].map(order)
    df = df.sort_values(["_r", "S", "i", "j"], ascending=[True, False, True, True], kind="mergesort")
    lead = ["gene_a", "gene_b", "relation", "statement", "kind", "source", "target", "canonical"]
    rest = [k for k in df.columns if k not in lead + ["_r", "i", "j"]]
    df = df[lead + rest].reset_index(drop=True)
    for r, k in df["relation"].value_counts().items():
        counts[r] = int(k)
    return df, counts


def _annotate(df, names):
    a = names[df["i"].to_numpy()]
    b = names[df["j"].to_numpy()]
    rel = df["relation"].to_numpy()
    words = {"low=>low": ("low", "low"), "low=>high": ("low", "high"),
             "high=>low": ("high", "low"), "high=>high": ("high", "high")}
    statement = [f"{x} {r} {y}" if r in ("equivalent", "opposite")
                 else f"{x} {words[r][0]} => {y} {words[r][1]}" for x, y, r in zip(a, b, rel)]
    kind = [KIND_OF[r] for r in rel]
    swap = rel == "low=>low"
    source = np.where(swap, b, a)
    target = np.where(swap, a, b)
    df["gene_a"] = a.astype(str)
    df["gene_b"] = b.astype(str)
    df["statement"] = statement
    df["kind"] = kind
    df["source"] = source.astype(str)
    df["target"] = target.astype(str)
    df["canonical"] = [f"{u} high => {v} high" if k == "HH" else st
                       for u, v, k, st in zip(source, target, kind, statement)]
    return df


def kind_counts(counts):
    out = dict.fromkeys(KINDS, 0)
    for r, k in counts.items():
        out[KIND_OF[r]] += k
    return out


def permutation_fdr(hi, lo, observed, n_perm=20, seed=0, **test):
    """FDR per relation kind: mean count after permuting each gene's samples
    independently, divided by the observed count (Sahoo et al., 2008)."""
    obs = kind_counts(observed)
    rng = np.random.default_rng(seed)
    n, G = hi.shape
    runs = []
    for _ in range(n_perm):
        order = np.argsort(rng.random((n, G)), axis=0)
        c = mine_relations(np.take_along_axis(hi, order, axis=0),
                           np.take_along_axis(lo, order, axis=0),
                           counts_only=True, **test)
        runs.append(kind_counts(c))
    perm = pd.DataFrame(runs, columns=KINDS) if runs else pd.DataFrame(columns=KINDS, dtype=float)
    rows = []
    for kind in KINDS + ["all"]:
        o = sum(obs.values()) if kind == "all" else obs[kind]
        p = perm.sum(axis=1) if kind == "all" else perm[kind]
        mean = float(p.mean()) if n_perm else np.nan
        sd = float(p.std(ddof=1)) if n_perm > 1 else np.nan
        fdr = min(1.0, mean / o) if (o and n_perm) else np.nan
        rows.append({"kind": kind, "observed": o, "permuted_mean": mean, "permuted_sd": sd, "fdr": fdr})
    return pd.DataFrame(rows)
