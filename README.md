# birgraph

Boolean implication networks of genes and gene clusters from an expression table.

Given a samples x genes table (Excel, CSV or TSV), birgraph binarises each gene with StepMiner, tests every gene pair for the six Boolean implication relationships of Sahoo et al. (2008), estimates false discovery rates by permutation, groups genes into clusters, derives cluster-level relations, and writes figures, interactive network viewers and tables.

The StepMiner threshold search and the binomial sparse-quadrant test are adapted from the BIRDNet code (https://github.com/tirtharajdash/BI-DNN). PyTorch is not needed.

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt
```

## Demo: UCI Mice Protein Expression data

The demo uses the public Mice Protein Expression dataset (Higuera et al., 2015): expression levels of 77 proteins and protein modifications in the nuclear fraction of the cerebral cortex of 38 control and 34 trisomic mice (a mouse model of Down syndrome), with 15 measurements of each protein per mouse, 1080 rows in total. The mice form eight classes by genotype, behaviour (context-shock or shock-context) and treatment (memantine or saline). The dataset has missing values and is licensed CC BY 4.0.

```
python examples/prepare_mice_protein.py
python run_birgraph.py --input examples/mice_protein/mice_protein_mouse_level.xlsx --group-col Genotype --group-order Control,Ts65Dn --margin-sd 0.25 --out results/mice_protein
```

The first script downloads the archive from the UCI repository, or reads a local copy given with `--source`, and writes two workbooks with log2 protein levels to `examples/mice_protein/`:

- `mice_protein_mouse_level.xlsx`: 72 rows, the mean of each mouse's 15 measurements;
- `mice_protein_replicates.xlsx`: all 1080 measurements.

It removes the `_N` (nuclear fraction) suffix from protein names and prints the birgraph command with the genotype labels found in the file. Reading the original .xls file needs xlrd, which is in `requirements.txt`.

Choices made for this dataset:

- Mouse level. The UCI description says each measurement can be treated as an independent sample. For Boolean implications, the replicate table counts every mouse 15 times; S and the group-test p-values grow with the number of rows, so replicate-level results look more significant than 72 mice support. The mouse-level table is the main demo. Running the same command on the replicate table shows the difference.
- Margin. The default margin of 0.5 comes from log2 microarray data (Sahoo et al., 2008). For these protein measurements, `--margin-sd 0.25` sets each protein's margin to a quarter of its robust standard deviation instead. Check `frac_intermediate` in `genes.csv` and `stepminer_examples.png`, and change the value if many proteins fail the dynamic-range filter.
- Sample size. With 72 mice, a quadrant can be called sparse only if more than 9 mice are expected in it (see Practical notes), so relations are found mainly between proteins whose high and low groups are not too unbalanced.
- Annotations. Treatment, Behavior and class are dropped with a warning unless one is given as `--group-col`; `--group-col class` compares the eight classes with Kruskal-Wallis tests.

## Input format

- One row per sample and one column per gene. The first column holds sample IDs (or name the column with `--sample-col`).
- An optional annotation column, for example `Group` with values Tumour and Normal, is given with `--group-col`. Other non-numeric columns are dropped with a warning. Missing values are allowed.
- For a genes x samples table, add `--genes-as-rows`.
- Values should be on a log2 scale: log2 intensities for microarrays, log2(TPM + 1) or log2(CPM + 1) for RNA-seq. For unlogged counts, add `--log2`, which applies log2(x + 1). The default margin of 0.5 assumes log2 units; `--margin-sd` does not depend on units.
- To restrict the analysis: `--gene-list genes.txt` (one gene per line), `--top-var N` (most variable genes), `--subset Tumour` (only samples of one group).

## Method

1. Threshold. For each gene, values are sorted and a one-step function is fitted by least squares; the threshold is placed at the step (StepMiner; Sahoo et al., 2007, 2008). `step_r2` in `genes.csv` is the fraction of variance explained by the step.
2. Discretisation. Values above threshold + margin are high, below threshold - margin low, and values in between intermediate. Intermediate and missing values are ignored for every pair involving that gene (Sahoo et al., 2008, with margin 0.5). The margin is `--margin` or, with `--margin-sd k`, k robust standard deviations (scaled median absolute deviation) of each gene.
3. Dynamic-range filter. Genes with more than `--max-intermediate` (default 2/3) of their samples in the intermediate band are not tested.
4. Pair test. For genes A and B, samples are counted in the four quadrants low/high x low/high. A quadrant is sparse when S > 3 and the error rate is below 0.1, with S = (expected - observed) / sqrt(expected), expected = n_A x n_B / total from the quadrant's row and column totals, and error rate = (observed / n_A + observed / n_B) / 2 (Sahoo et al., 2008). One sparse quadrant gives an asymmetric relation; two diagonally opposite sparse quadrants give equivalent or opposite. Pairs in which either gene is low or high in fewer than 5% of the tested samples are skipped (`--min-frac`). With `--method binomial`, a quadrant is sparse instead when a one-sided binomial test gives p < 1e-6 and it holds at most 5% of the samples, as in BIRDNet.
5. False discovery rate. The test is repeated `--n-perm` times (default 20) after shuffling each gene's samples independently. The FDR of a relation type is the mean permuted count divided by the observed count (Sahoo et al., 2008). An FDR of 0 means that no relation of that type appeared in any permutation.
6. Clusters. `--clusters components` (default) uses connected components of the graph of equivalent gene pairs; `louvain` uses Louvain communities of the same graph (Blondel et al., 2008); `map` reads a two-column gene, cluster table given with `--cluster-map`.
7. Cluster relations. For two clusters with at least `--min-cluster-size` genes, the most frequent gene-level relation between their members is kept if it covers at least `--min-support` (default 0.5) of all member pairs. BoNE also summarises Boolean implications between gene clusters (Sahoo et al., 2021); this support rule is specific to birgraph.
8. Group statistics (with `--group-col`). Per gene: mean and fraction of high samples in each group. For two groups, the difference (second minus first level of `--group-order`) with a Mann-Whitney U test; for more groups, a Kruskal-Wallis test. q-values use the Benjamini-Hochberg procedure (Benjamini and Hochberg, 1995).

## Relation types

| Relation | Sparse quadrant | Network edge |
|---|---|---|
| A low => B low | A low, B high | blue arrow from B to A (the same as B high => A high) |
| A high => B high | A high, B low | blue arrow from A to B |
| A high => B low | both high | pink dashed line (the same as B high => A low) |
| A low => B high | both low | orange dotted line (the same as B low => A high) |
| A equivalent B | A low B high, and A high B low | green line |
| A opposite B | both low, and both high | red-orange line |

## Outputs

| File | Content |
|---|---|
| `birgraph_report.xlsx` | Sheets summary, genes, relations, fdr, clusters, cluster_relations. Tables longer than 200,000 rows are truncated here; the CSV files are complete. |
| `relations.csv` | One row per related gene pair: relation, statement, S, error rate, binomial p, quadrant counts, Pearson r |
| `genes.csv` | Threshold, margin, step R2, fractions low/intermediate/high, filter result, cluster, relations by type, group statistics |
| `cluster_relations.csv` | Cluster-level relations with support and purity |
| `gene_network.png`, `.pdf` | Gene network coloured by cluster |
| `gene_network_by_group.png`, `.pdf` | Gene network coloured by the group difference |
| `cluster_network.png`, `.pdf` and `cluster_network_by_group.png`, `.pdf` | Cluster networks; node labels give the cluster size |
| `gene_network.html`, `cluster_network.html` | Standalone viewers: zoom, filter relation types, search, click a node to list its relations |
| `gene_network.graphml`, `cluster_network.graphml` | For Cytoscape or Gephi |
| `relation_counts.png` | Observed and permuted counts per relation type, with FDR |
| `relation_examples.png` | Strongest example of each relation type, between clusters where possible |
| `stepminer_examples.png` | Example thresholds, including a gene that fails the dynamic-range filter |
| `config.json` | All settings of the run |

Network figures show at most the `--max-plot-edges` (default 3000) strongest relations, and gene labels are drawn for networks of up to 200 genes. The HTML viewers need no internet connection.

## Practical notes

- Sample size. An empty quadrant reaches S > 3 only when its expected count exceeds 9. For two genes that are each high in 30% of the tested samples, the both-high quadrant can only be called sparse with more than 100 samples. Small cohorts therefore yield few relations, and mainly strong ones.
- Mixed tissues. Relations mined from tumour and normal samples together include tissue-type differences. Use `--subset Tumour` to study relations within tumours, and compare with a run on normal samples.
- Replicates. Technical replicates are not independent samples; average them per biological sample, as the demo does.
- Batch effects can create spurious implications; correct or remove them before the analysis.
- Run time grows with the square of the number of genes. On a single CPU core in a test environment, 400 samples x 2,000 genes took about 20 s end to end (2 permutations), and 5,000 genes took about 4 s per mining pass, repeated once for each permutation. Starting with `--top-var 5000` is reasonable.
- With `--clusters components`, chains of equivalences can merge unrelated modules into one large cluster on big datasets. If the largest cluster is very large, try `--clusters louvain`.

## Python use

```python
from birgraph import assign_clusters, cluster_relations, fit_thresholds, load_expression, mine_relations

expr, groups = load_expression("data.xlsx", group_col="Group")
genes, hi, lo = fit_thresholds(expr)
keep = genes["passes_dynamic_range"].to_numpy()
names = genes.loc[keep, "gene"].tolist()
relations, counts = mine_relations(hi[:, keep], lo[:, keep], names)
labels, clusters = assign_clusters(names, relations)
cluster_edges = cluster_relations(relations, labels, clusters)
```

## Tests

`python tests/test_birgraph.py` (or `pytest tests`) checks the StepMiner search against the original quadratic search, recovery of planted relations and clusters in synthetic data (`tests/synthetic.py`), that `--margin-sd` gives the same relations after rescaling each gene, the network layout, the preparation of the mice data on a mock table with the same layout, and a full command-line run.

## Package layout

```
run_birgraph.py                  command-line pipeline
birgraph/io.py                   reading tables, gene selection
birgraph/stepminer.py            thresholds, margins, discretisation
birgraph/mining.py               pairwise relations, permutation FDR
birgraph/clusters.py             gene clusters, cluster-level relations
birgraph/groups.py               group statistics
birgraph/network.py              graphs, summaries, layout
birgraph/plots.py                figures
birgraph/viewer.py               interactive HTML viewer
birgraph/report.py               Excel report
examples/prepare_mice_protein.py download and preparation of the demo data
tests/                           checks and synthetic test data
```

## References

- Sahoo D, Dill DL, Tibshirani R, Plevritis SK. Extracting binary signals from microarray time-course data. Nucleic Acids Research 35(11):3705-3712, 2007. https://doi.org/10.1093/nar/gkm284
- Sahoo D, Dill DL, Gentles AJ, Tibshirani R, Plevritis SK. Boolean implication networks derived from large scale, whole genome microarray datasets. Genome Biology 9:R157, 2008. https://doi.org/10.1186/gb-2008-9-10-r157
- Sahoo D. The power of Boolean implication networks. Frontiers in Physiology 3:276, 2012. https://doi.org/10.3389/fphys.2012.00276
- Sahoo D, et al. Artificial intelligence guided discovery of a barrier-protective therapy in inflammatory bowel disease. Nature Communications 12:4246, 2021. https://doi.org/10.1038/s41467-021-24470-5
- Higuera C, Gardiner KJ, Cios KJ. Self-organizing feature maps identify proteins critical to learning in a mouse model of Down syndrome. PLoS ONE 10(6):e0129126, 2015. https://doi.org/10.1371/journal.pone.0129126
- Higuera C, Gardiner K, Cios K. Mice Protein Expression [Dataset]. UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C50S3Z
- Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment P10008, 2008.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 57(1):289-300, 1995.
