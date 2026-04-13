# Design: Risk-Only NMF Primary Clustering

**Date:** 2026-04-13  
**Project:** ECP77594 — Critical-but-Fragile (final_project_v2)  
**Status:** Approved

---

## Problem statement

The current analysis runs NMF topic modelling on all 9,461 load-bearing packages, then maps the 247 risk-universe packages into those corpus-shaped clusters. This means the cluster structure is driven by what is *common* in the full corpus (React, Vue, Babel, etc.), not by what is actually fragile and critical. The result: the most fragile "cluster" is a residual category (Topic 0, "Generic/legacy infrastructure") that NMF could not assign elsewhere because those packages had uninformative descriptions relative to the full-corpus vocabulary. The cluster-level findings in Table 1 and Figure 2 are artefacts of fitting topics to the wrong population.

The fix: identify the risk universe first (unchanged), then cluster *only those 247 packages* with a separate NMF pass. The full-corpus NMF is retained as descriptive context.

---

## Architecture

```
Score (Stage 2)                     → criticality, fragility, in_risk_universe
Full-corpus NMF (Stage 3b)          → topic_id_corpus, topic_label_corpus on all 9,461
Risk-only NMF (Stage 3c, NEW)       → topic_id_risk, topic_label_risk on the 247
Risk cluster analysis (Stage 4)     → uses topic_id_risk (rewritten)
Threshold sensitivity (Stage 4d)    → uses corpus labels, unchanged
Figures (Stage 8)                   → Fig 1 updated, Fig 2 rebuilt
```

---

## Stage 3b — full-corpus NMF (unchanged)

- Input: all 9,461 packages
- k=25, RANDOM_SEED=42
- Assigns `topic_id_corpus` and `topic_label_corpus` to every row
- Used only in: threshold sensitivity (Stage 4d), descriptive printout
- Column rename: current `topic_id` → `topic_id_corpus`, `topic_label` → `topic_label_corpus`

---

## Stage 3c — risk-only NMF (new)

**Input:** `df[df["in_risk_universe"]]` (247 rows at 75th-percentile threshold)

**TF-IDF:** Re-fitted on risk-set descriptions only (same stop words, `max_features=2000`, `min_df=2`, `max_df=0.8`). Vocabulary constrained to what appears in the risk set — avoids bleeding in npm-ecosystem signal from the full corpus.

**Topic count selection:**
- Sweep k=5..8, print reconstruction error and top-4 keywords per topic per k
- `N_RISK_TOPICS = 6` as the default constant (user adjusts by inspecting sweep output and updating the constant)
- Same `RANDOM_SEED=42`

**Output columns added to `df` (NaN for non-risk rows):**
- `topic_id_risk` (int, NaN for non-risk)
- `topic_label_risk` (str, NaN for non-risk)

**Human labels:**
- New `HUMAN_RISK_TOPIC_LABELS` dict with placeholder `"Risk cluster N"` entries
- User overwrites with descriptive labels after inspecting first-run output
- Same transparency approach as full-corpus: auto-keywords printed, human labels used downstream

---

## Stage 4 — cluster analysis (rewritten)

`analyse_clusters()` now receives `df[df["in_risk_universe"]]` and groups by `topic_id_risk`.

Per-cluster statistics:
- n_packages
- mean_fragility, median_fragility
- mean_criticality
- pct_bus_factor_1
- mean_days_since_release
- funding_gap_eur (n_packages × cost_per_pkg)

Sorted by mean_fragility descending.

`inspect_top_cluster()` and `compute_funding_gap()` updated to use `topic_id_risk` / `topic_label_risk`.

---

## Stage 4d — threshold sensitivity (unchanged)

Uses full-corpus cluster labels (k=25) to check cross-threshold stability. Rationale: threshold sensitivity is a stability check; it needs a fixed cluster structure to be interpretable. Re-fitting risk-only NMF at each threshold would produce incomparable cluster labels.

A one-line note added to the printed output:
> "Cluster labels here are full-corpus NMF (k=25); risk-only clusters not re-fit per threshold."

---

## Figures

### Figure 1 — C–F scatter (updated)

- Background: all 9,461 points in light grey (alpha=0.1, no cluster colouring)
- Risk-universe 247 points: coloured by `topic_id_risk`, larger markers, black edges
- Legend shows risk-cluster labels with per-cluster counts
- Threshold dashed lines unchanged

### Figure 2 — cluster comparison (rebuilt)

- Panel (a): mean fragility by risk cluster (unchanged structure, new data)
- Panel (b): **funding gap by risk cluster** (replaces risk-concentration bar chart — risk concentration is 100% by construction for all risk-only clusters, so the old panel is uninformative)

### Figure 0b — risk-NMF k sweep (new, small)

- Analogous to Fig 0 (full-corpus sweep)
- k=5..8 reconstruction error, selected k marked
- Saved as `outputs/fig0b_risk_topic_selection.png`

### Figures 0, 3 (bootstrap) — unchanged

---

## Outputs

| File | Change |
|------|--------|
| `outputs/packages_scored.csv` | Gains `topic_id_risk`, `topic_label_risk`; `topic_id`/`topic_label` renamed to `topic_id_corpus`/`topic_label_corpus` |
| `outputs/risk_cluster_summary.csv` | New — primary cluster findings from risk-only NMF |
| `outputs/cluster_summary.csv` | Kept — full-corpus descriptive context |
| `outputs/funding_gap_by_cluster.csv` | Rebuilt from risk-only clusters |
| `outputs/fig0b_risk_topic_selection.png` | New |
| `outputs/fig1_scatter.png` | Updated (grey background, risk recoloured) |
| `outputs/fig2_clusters.png` | Rebuilt (risk clusters, panel b = funding gap) |

---

## Notebook (analysis.ipynb)

Same structural changes mirrored:
- Stage 3b cell block: column renames
- New Stage 3c cell block inserted between 3b and 4
- Stage 4 cell block: updated to use risk labels
- Figure cells updated to match

---

## What does NOT change

- `preprocess.py` — no changes
- Scoring (Stage 2) — no changes
- C–F correlation check (Stage 2b) — no changes
- Full-corpus NMF itself (Stage 3b) — logic unchanged, column names change
- LASSO validation (Stage 4c) — no changes
- Threshold sensitivity (Stage 4d) — no changes
- Bootstrap (Stage 7) — no changes
- `HUMAN_TOPIC_LABELS` (full-corpus, k=25) — no changes

---

## Decisions and trade-offs

| Decision | Rationale |
|----------|-----------|
| Re-fit TF-IDF on risk set | Corpus vocabulary is npm-heavy; risk set has different description distribution. Re-fitting ensures topic terms reflect what distinguishes risk packages from each other, not from the full corpus. |
| Default N_RISK_TOPICS=6 | 247 packages ÷ 6 ≈ 41 per cluster — enough for stable statistics. k=3 is too coarse; k=8 averages 31 per cluster, borderline. User adjusts after inspecting sweep. |
| Keep full-corpus NMF | Threshold sensitivity requires stable cluster labels across thresholds. Corpus clusters serve this purpose. Also retains the k=25 interpretability investment. |
| Panel (b) → funding gap | With risk-only clusters, all members are in the risk universe, so risk-concentration is 100% everywhere — the old panel becomes degenerate. Funding gap is the economically meaningful per-cluster quantity. |
| Grey background in Fig 1 | Removes visual ambiguity between corpus-cluster colouring and risk-cluster colouring. Background is context; risk points are the finding. |
