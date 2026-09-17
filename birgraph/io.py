import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path, sheet=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def load_expression(path, sheet=None, sample_col=None, group_col=None,
                    genes_as_rows=False, log2=False):
    """Read a samples x genes table.

    The first column (or `sample_col`) holds sample IDs. `group_col` is an
    optional annotation column (for example Tumour/Normal). With
    `genes_as_rows`, the first column holds gene IDs and the group
    annotation, if any, is a row named `group_col`.
    """
    df = read_table(path, sheet)
    if genes_as_rows:
        df = df.set_index(df.columns[0]).T
        df.index.name = "SampleID"
        df = df.reset_index()
    sample_col = sample_col or df.columns[0]
    df = df.set_index(sample_col)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if df.index.duplicated().any():
        warnings.warn("duplicated sample IDs found")

    groups = None
    if group_col is not None:
        if group_col not in df.columns:
            raise KeyError(f"group column '{group_col}' not found")
        groups = df.pop(group_col).astype(str)

    expr = df.apply(pd.to_numeric, errors="coerce")
    empty = expr.columns[expr.isna().all()]
    if len(empty):
        warnings.warn(f"dropping {len(empty)} non-numeric columns, e.g. {list(empty)[:5]}")
        expr = expr.drop(columns=empty)

    if log2:
        if np.nanmin(expr.to_numpy()) < 0:
            raise ValueError("negative values found; log2(x + 1) is not applicable")
        expr = np.log2(expr + 1.0)
    elif np.nanmax(expr.to_numpy()) > 100:
        warnings.warn("values above 100 found; the data may not be log-transformed (see --log2)")
    return expr, groups


def read_gene_list(path):
    lines = Path(path).read_text().splitlines()
    return [g.strip() for g in lines if g.strip()]


def select_genes(expr, gene_list=None, top_var=None):
    if gene_list is not None:
        keep = [g for g in dict.fromkeys(gene_list) if g in expr.columns]
        if len(keep) < len(set(gene_list)):
            warnings.warn(f"{len(set(gene_list)) - len(keep)} genes from the list are not in the data")
        expr = expr[keep]
    if top_var is not None and expr.shape[1] > top_var:
        var = expr.var(axis=0, skipna=True).sort_values(ascending=False, kind="mergesort")
        expr = expr[var.index[:top_var]]
    return expr


def load_cluster_map(path, sheet=None):
    """Two-column table: gene, cluster."""
    df = read_table(path, sheet)
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
