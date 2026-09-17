import warnings

import numpy as np
import pandas as pd


def stepminer_threshold(x):
    """One-step StepMiner fit to sorted values.

    Same split search as `bir.stepmine` in BIRDNet, computed in O(n) with
    cumulative sums. Returns (threshold, r2), where r2 is the fraction of
    variance explained by the fitted step.
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    if n < 4:
        return (float(np.median(x)) if n else np.nan), 0.0
    xc = x - x.mean()
    c1 = np.cumsum(xc)
    c2 = np.cumsum(xc * xc)
    k = np.arange(1, n - 1)
    n_lo = k + 1.0
    n_hi = n - k - 1.0
    sse = (c2[k] - c1[k] ** 2 / n_lo) + (c2[-1] - c2[k] - (c1[-1] - c1[k]) ** 2 / n_hi)
    b = int(np.argmin(sse))
    kb = k[b]
    sst = c2[-1]
    r2 = float(1.0 - sse[b] / sst) if sst > 1e-12 else 0.0
    return float(0.5 * (x[kb] + x[kb + 1])), r2


def robust_sd(X):
    """Per-column scaled median absolute deviation, or the standard deviation where the MAD is zero."""
    X = np.asarray(X, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(X - med), axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
    return np.where(mad > 0, mad, sd)


def discretise(X, thresholds, margin=0.5):
    """High above t + margin, low below t - margin, otherwise intermediate (ignored).

    margin is a scalar or one value per gene; missing values count as intermediate.
    """
    with np.errstate(invalid="ignore"):
        hi = X > thresholds + margin
        lo = X < thresholds - margin
    return hi, lo


def fit_thresholds(expr, margin=0.5, max_intermediate=2 / 3, margin_sd=None):
    """Per-gene thresholds, margins and Boolean states.

    The margin is `margin` for every gene or, with `margin_sd`, that many
    robust standard deviations of each gene. Genes with more than
    `max_intermediate` of their values in the intermediate band fail the
    dynamic-range filter.
    """
    X = expr.to_numpy(dtype=float)
    thr = np.empty(X.shape[1])
    r2 = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        thr[j], r2[j] = stepminer_threshold(X[:, j])
    margins = margin_sd * robust_sd(X) if margin_sd is not None else np.full(X.shape[1], float(margin))
    hi, lo = discretise(X, thr, margins)
    n_obs = np.isfinite(X).sum(axis=0)
    denom = np.maximum(n_obs, 1)
    table = pd.DataFrame({
        "gene": expr.columns,
        "threshold": thr,
        "margin": margins,
        "step_r2": r2,
        "n_obs": n_obs,
        "frac_low": lo.sum(axis=0) / denom,
        "frac_intermediate": 1.0 - (hi.sum(axis=0) + lo.sum(axis=0)) / denom,
        "frac_high": hi.sum(axis=0) / denom,
    })
    table["passes_dynamic_range"] = table["frac_intermediate"] <= max_intermediate
    return table, hi, lo
