# Critical-but-Fragile: Mapping Fragility Across Functional Categories of Open Source Infrastructure

ECP77594 — Computational Methods in Economics — Final Project — Sarah Novotny

## Repository layout

```
final_project_v2/
├── README.md                  this file
├── requirements.txt           pinned Python dependencies
├── proposal.md                original project proposal (+ change log)
├── PROJECT_OVERVIEW.md        operational decisions log (companion to the report)
├── preprocess.py              raw_data/ → data/   (run once)
├── analysis.py                data/ → outputs/    (script form, end-to-end)
├── analysis.ipynb             data/ → outputs/    (notebook form, identical logic)
├── raw_data/
│   ├── README.md              how to obtain the inputs
│   ├── libraries-1.6.0-…/     symlink to the 159 GB Libraries.io snapshot
│   └── CNECT_OpenSourceStudy…pdf  EU Commission OSS study (reference)
├── data/                      analysis-ready CSVs (built by preprocess.py)
│   ├── packages_loadbearing.csv      9,461 rows × 12 cols
│   └── eurostat_gva_2022.csv         31 countries × 2 cols
└── outputs/                   results (built by analysis.py / .ipynb)
    ├── packages_scored.csv           9,461 rows × 18 cols (incl. topic_id_risk, topic_label_risk)
    ├── cluster_summary.csv           full-corpus NMF (k=25) cluster table — descriptive context
    ├── risk_cluster_summary.csv      risk-only NMF (k=5) cluster table — primary findings
    ├── funding_gap_by_cluster.csv    per-cluster funding gap (risk-only clusters)
    ├── fig0_topic_selection.png      full-corpus k sweep
    ├── fig0b_risk_topic_selection.png  risk-only k sweep (k=3–8)
    ├── fig1_scatter.png              C–F scatter, risk points coloured by risk cluster
    ├── fig2_clusters.png             fragility ranking + funding gap by risk cluster
    └── fig3_bootstrap.png            bootstrap distributions
```

## Reproducing the results

```bash
# 1. Create a clean Python 3.12 environment and install dependencies.
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Obtain the raw Libraries.io snapshot (see raw_data/README.md) and
#    place it (or a symlink to it) at raw_data/libraries-1.6.0-2020-01-12/

# 3. Build the analysis-ready CSVs. This is the only step that touches
#    the 25 GB raw CSV and the only step that calls a remote API.
python preprocess.py
#    → writes data/packages_loadbearing.csv  (~5 min)
#    → writes data/eurostat_gva_2022.csv     (cached after first run)

# 4. Reproduce all results, tables, and figures.
python analysis.py
#    → writes outputs/*.csv and outputs/fig*.png  (~3 min, including 1,000 bootstrap replicates)

# Equivalent notebook form:
jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
```

Random seed `RANDOM_SEED = 42` is set on the NMF, the LASSO CV split, and the bootstrap RNG, so every run produces byte-identical numbers given the same input CSVs.

## Headline numbers

| Metric | Value |
|---|---|
| Load-bearing packages (≥ 100 dependent projects) | 9,461 |
| Risk universe (≥ 75th percentile on both criticality and fragility) | 247 |
| Criticality–fragility correlation (Pearson r) | −0.31 |
| Topic model — full corpus | k=25 NMF on all 9,461 packages (corpus context, threshold sensitivity) |
| Topic model — risk universe | k=5 NMF on 247 risk packages only (primary findings) |
| Most fragile risk cluster | Apache / Java Commons legacy infrastructure (27 pkgs, F̄=0.721, 74% bus-factor-1) |
| Largest risk cluster | Microsoft / .NET + npm ecosystem (127 pkgs, EUR 11.4M gap) |
| EU J62/J63 Gross Value Added (2022, sum of member states) | EUR 466,568 M |
| Maintenance funding gap (gross lower bound; 2 FTE × EUR 45 K per risk-universe package) | ≥ EUR 22.2 M [95 % CI: 19.7, 24.8 — sampling uncertainty only] |
| Bootstrap replicates | 1,000 |

## Method, scope, and limitations

The full methodological narrative lives in two companion documents:

- **`proposal.md`** — the original research question, data sources, and methods, plus a section recording how the plan changed during execution.
- **`PROJECT_OVERVIEW.md`** — a 14-section operational log of every decision made during the analysis, including the CSV off-by-one fix, the choice of NMF over LDA, the topic-count selection at k = 25, the LASSO circularity caveat, and the funding-gap assumptions.

The final report PDF (still in draft as of submission) summarises both into a 2,000–2,300 word academic-style paper.

## Data sources

1. **Libraries.io v1.6.0** (Katz, 2020 — Zenodo): https://doi.org/10.5281/zenodo.3626071
2. **Eurostat National Accounts** (`nama_10_a64`, J62_J63, B1G, CP_MEUR), retrieved via DBnomics.
3. **EU Commission Open Source Study** (Blind et al., 2021 — DG CNECT): used as background reference, not as input.
