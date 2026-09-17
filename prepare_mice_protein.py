"""Download the UCI Mice Protein Expression dataset and prepare it for birgraph.

Source: Higuera C, Gardiner K, Cios K. Mice Protein Expression [Dataset]. UCI
Machine Learning Repository, 2015. https://doi.org/10.24432/C50S3Z (CC BY 4.0).
"""

import argparse
import io
import os
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PAGE = "https://archive.ics.uci.edu/dataset/342/mice+protein+expression"
URL = "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip"
ANNOTATIONS = ["Genotype", "Treatment", "Behavior", "class"]
NOTES = [
    "UCI Mice Protein Expression dataset, prepared for birgraph with examples/prepare_mice_protein.py.",
    "Source: Higuera C, Gardiner K, Cios K. Mice Protein Expression [Dataset]. UCI Machine Learning Repository, 2015. "
    "https://doi.org/10.24432/C50S3Z. Licence: CC BY 4.0.",
    "Study: Higuera C, Gardiner KJ, Cios KJ. Self-organizing feature maps identify proteins critical to learning in a "
    "mouse model of Down syndrome. PLoS ONE 10(6):e0129126, 2015.",
    "77 proteins and protein modifications measured in the nuclear fraction of cortex of 38 control and 34 trisomic "
    "mice, 15 measurements per protein per mouse.",
    "Values are log2 of the published expression levels; non-positive values are treated as missing. "
    "The _N suffix (nuclear fraction) is removed from protein names.",
]


def load_raw(source, folder):
    """Raw table from a local zip or Excel file, downloading the UCI archive if no file is given."""
    if source is None:
        source = folder / Path(URL).name
        if not source.exists():
            print(f"Downloading {URL}")
            try:
                urllib.request.urlretrieve(URL, source)
            except OSError as err:
                source.unlink(missing_ok=True)
                sys.exit(f"Download failed ({err}). Download the zip file from {PAGE} and rerun with --source <file>.")
    source = Path(source)
    try:
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
                if not names:
                    sys.exit(f"No Excel file found in {source}")
                return pd.read_excel(io.BytesIO(zf.read(names[0])))
        return pd.read_excel(source)
    except ImportError as err:
        sys.exit(f"{err}\nReading the .xls file needs xlrd: pip install xlrd")


def prepare(raw):
    """Return (replicates, mice): log2 levels of every measurement, and their mean per mouse.

    MouseID values have the form <mouse>_<replicate>.
    """
    raw = raw.rename(columns=lambda c: str(c).strip())
    if "MouseID" not in raw.columns:
        raise ValueError("column MouseID not found")
    raw = raw[raw["MouseID"].notna()].reset_index(drop=True)
    annot = [c for c in ANNOTATIONS if c in raw.columns]
    proteins = [c for c in raw.columns if c not in ["MouseID", *annot]]
    values = raw[proteins].apply(pd.to_numeric, errors="coerce")
    if all(c.upper().endswith("_N") for c in proteins):
        values.columns = [c[:-2] for c in proteins]
    n_bad = int((values <= 0).sum().sum())
    if n_bad:
        warnings.warn(f"{n_bad} non-positive values treated as missing")
    log2 = np.log2(values.where(values > 0))

    ids = raw["MouseID"].astype(str).str.strip()
    replicates = pd.concat([ids.rename("SampleID"), raw[annot], log2], axis=1)
    mouse = ids.str.extract(r"^(.+)_(\d+)$")[0]
    if mouse.isna().any():
        raise ValueError("MouseID values are not all of the form <mouse>_<replicate>")
    mouse = mouse.rename("SampleID")
    grouped = raw[annot].groupby(mouse, sort=False)
    varying = grouped.nunique().gt(1).any(axis=1)
    if varying.any():
        warnings.warn(f"annotations differ between replicates of {int(varying.sum())} mice; the first value is used")
    mice = pd.concat([grouped.first(), log2.groupby(mouse, sort=False).mean()], axis=1).reset_index()
    return replicates, mice


def write(table, path, notes):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        table.round(5).to_excel(xw, sheet_name="expression", index=False)
        pd.DataFrame({"README": notes}).to_excel(xw, sheet_name="README", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download and prepare the UCI Mice Protein Expression dataset for birgraph.")
    ap.add_argument("--source", default=None, help="local copy of the UCI zip file or Data_Cortex_Nuclear.xls")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "mice_protein"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    replicates, mice = prepare(load_raw(args.source, out))
    mouse_file = out / "mice_protein_mouse_level.xlsx"
    replicate_file = out / "mice_protein_replicates.xlsx"
    write(mice, mouse_file, NOTES + ["Mouse level: one row per mouse, the mean of its log2 replicate measurements."])
    write(replicates, replicate_file,
          NOTES + ["Replicate level: one row per measurement; SampleID is the original MouseID <mouse>_<replicate>."])
    n_proteins = mice.shape[1] - 1 - sum(c in mice.columns for c in ANNOTATIONS)
    print(f"{len(mice)} mice, {len(replicates)} measurements, {n_proteins} proteins")
    print(f"Wrote {os.path.relpath(mouse_file)}\nWrote {os.path.relpath(replicate_file)}")
    group = ""
    if "Genotype" in mice.columns:
        levels = sorted(mice["Genotype"].dropna().astype(str).unique(),
                        key=lambda v: (not v.lower().startswith("control"), v))
        group = f" --group-col Genotype --group-order {','.join(levels)}"
    print("\nRun birgraph on the mouse-level table:")
    print(f"python run_birgraph.py --input {os.path.relpath(mouse_file)}{group} --margin-sd 0.25 "
          f"--out results/mice_protein")


if __name__ == "__main__":
    main()
