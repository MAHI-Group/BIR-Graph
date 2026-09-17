import numpy as np
import pandas as pd
from scipy import stats


def group_statistics(expr, hi, groups, order=None):
    """Per-gene group summaries.

    Two groups: mean difference (second minus first level) and Mann-Whitney U
    test. More groups: Kruskal-Wallis test. q-values use Benjamini-Hochberg.
    """
    levels = list(order) if order else sorted(groups.unique())
    missing = set(levels) - set(groups.unique())
    if missing:
        raise ValueError(f"group levels not found: {sorted(missing)}; available: {sorted(groups.unique())}")
    X = expr.to_numpy(dtype=float)
    masks = [(groups == lv).to_numpy() for lv in levels]
    out = pd.DataFrame({"gene": expr.columns})
    for lv, m in zip(levels, masks):
        out[f"mean_{lv}"] = np.nanmean(X[m], axis=0)
        out[f"frac_high_{lv}"] = hi[m].mean(axis=0)
    with np.errstate(all="ignore"):
        if len(levels) == 2:
            out["delta"] = out[f"mean_{levels[1]}"] - out[f"mean_{levels[0]}"]
            p = stats.mannwhitneyu(X[masks[1]], X[masks[0]], axis=0, nan_policy="omit").pvalue
        else:
            p = stats.kruskal(*[X[m] for m in masks], axis=0, nan_policy="omit").pvalue
    p = np.asarray(p, dtype=float)
    out["p_value"] = p
    out["q_value"] = stats.false_discovery_control(np.where(np.isfinite(p), p, 1.0), method="bh")
    return out, levels
