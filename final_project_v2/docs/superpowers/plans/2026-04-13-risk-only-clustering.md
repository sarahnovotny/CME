# Risk-Only NMF Primary Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the analysis so NMF clusters the 247 risk-universe packages on their own, not as a subset of the 9,461-package corpus, making cluster labels and the cluster-fragility ranking reflect what actually distinguishes at-risk packages from each other.

**Architecture:** Two-pass NMF: full-corpus NMF (k=25) is retained for threshold-sensitivity analysis and renamed to `topic_id_corpus`/`topic_label_corpus`; a new Stage 3c fits a separate NMF on risk-universe descriptions only (k swept 5–8, default k=6), producing `topic_id_risk`/`topic_label_risk`. Stage 4 onwards uses risk labels. Figure 1 gets a grey background + risk points recoloured by risk labels. Figure 2 panel (b) changes from risk-concentration bars (degenerate at 100%) to funding gap by risk cluster.

**Tech Stack:** Python 3.12, pandas, scikit-learn (TfidfVectorizer, NMF), matplotlib. Run via `source /path/to/python3.12-venv/bin/activate && python analysis.py`. Notebook `analysis.ipynb` mirrors `analysis.py` and is updated last.

**Activate venv for all commands:** `source /Users/sarahnovotny/Github/python3.12-venv/bin/activate`

---

## File map

| File | Change |
|------|--------|
| `analysis.py` | All logic changes — 8 tasks |
| `analysis.ipynb` | Mirror all changes from analysis.py |
| `outputs/packages_scored.csv` | Gains `topic_id_risk`, `topic_label_risk`; `topic_id`/`topic_label` → `topic_id_corpus`/`topic_label_corpus` |
| `outputs/risk_cluster_summary.csv` | New file |
| `outputs/funding_gap_by_cluster.csv` | Rebuilt from risk clusters |
| `outputs/fig0b_risk_topic_selection.png` | New |
| `outputs/fig1_scatter.png` | Updated |
| `outputs/fig2_clusters.png` | Rebuilt |

---

## Task 1: Add constants and rename corpus columns in `topic_model()`

**Files:**
- Modify: `analysis.py:69-102` (constants block)
- Modify: `analysis.py:356-423` (`topic_model()`)

- [ ] **Step 1: Add `N_RISK_TOPICS` and `HUMAN_RISK_TOPIC_LABELS` constants**

In `analysis.py`, after line 69 (`N_TOPICS = 25`), add:

```python
N_RISK_TOPICS = 6         # topics for risk-only NMF; adjust after inspecting sweep output
```

After the `HUMAN_TOPIC_LABELS` dict (after line 102), add:

```python
# Human-readable labels for the risk-only NMF clusters.
# Placeholder values — overwrite after inspecting the first-run sweep output
# and the auto-generated keywords printed by topic_model_risk().
HUMAN_RISK_TOPIC_LABELS = {
    0: "Risk cluster 0",
    1: "Risk cluster 1",
    2: "Risk cluster 2",
    3: "Risk cluster 3",
    4: "Risk cluster 4",
    5: "Risk cluster 5",
}
```

- [ ] **Step 2: Rename corpus columns in `topic_model()`**

In `topic_model()`, find the two lines that assign `topic_id` and `topic_label` to `df` (lines 386 and 408) and rename them:

```python
# line 386 — change:
df["topic_id"] = W.argmax(axis=1)
df["topic_weight"] = W.max(axis=1)
# to:
df["topic_id_corpus"] = W.argmax(axis=1)
df["topic_weight_corpus"] = W.max(axis=1)
```

```python
# line 408 — change:
df["topic_label"] = df["topic_id"].map(topic_labels)
# to:
df["topic_label_corpus"] = df["topic_id_corpus"].map(topic_labels)
```

Also update the loop body that references `df["topic_id"]` on line 397 and 398:

```python
# line 397-398 — change:
        n_pkgs = (df["topic_id"] == topic_idx).sum()
        n_risk = ((df["topic_id"] == topic_idx) & df["in_risk_universe"]).sum()
# to:
        n_pkgs = (df["topic_id_corpus"] == topic_idx).sum()
        n_risk = ((df["topic_id_corpus"] == topic_idx) & df["in_risk_universe"]).sum()
```

And the summary print loop at line 412-413:

```python
# change:
    for tid in sorted(topic_labels.keys()):
        n = (df["topic_id"] == tid).sum()
# to:
    for tid in sorted(topic_labels.keys()):
        n = (df["topic_id_corpus"] == tid).sum()
```

And line 416-418 (largest cluster check):

```python
# change:
    largest = df["topic_id"].value_counts().iloc[0]
# to:
    largest = df["topic_id_corpus"].value_counts().iloc[0]
```

- [ ] **Step 3: Verify the rename doesn't break the script up to Stage 3b**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
print('columns with corpus:', [c for c in df.columns if 'topic' in c])
assert 'topic_id_corpus' in df.columns, 'rename failed'
assert 'topic_id' not in df.columns, 'old column still present'
print('PASS')
"
```

Expected: prints `columns with corpus: ['topic_id_corpus', 'topic_weight_corpus', 'topic_label_corpus']` and `PASS`.

- [ ] **Step 4: Commit**

```bash
git add analysis.py
git commit -m "refactor: rename topic_id/label → topic_id_corpus/label_corpus for two-pass NMF"
```

---

## Task 2: Add `topic_model_risk()` — Stage 3c

**Files:**
- Modify: `analysis.py` — insert new function after `topic_model()` (after line 423)

- [ ] **Step 1: Insert `topic_model_risk()` after `topic_model()`**

Insert the following function into `analysis.py` after the closing `return` of `topic_model()` (after line 423), before the `# ── Stage 4` comment:

```python
# ── Stage 3c: Risk-only topic modelling ───────────────────────────────

def topic_model_risk(df):
    """Run NMF topic modelling on risk-universe package descriptions only.

    TF-IDF is re-fitted on the risk set so vocabulary reflects what
    distinguishes at-risk packages from each other, not from the full corpus.
    A k sweep (5..8) is printed for manual inspection; N_RISK_TOPICS is used
    for the final fit. Update HUMAN_RISK_TOPIC_LABELS after inspecting the
    printed auto-keywords.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 3c: Risk-only topic modelling (NMF k sweep 5–8, selected k={N_RISK_TOPICS})")
    print("=" * 70)

    risk_df = df[df["in_risk_universe"]].copy()
    print(f"  Risk universe: {len(risk_df)} packages")

    texts = _clean_descriptions(risk_df["Description"].fillna("").astype(str))
    stop_words = _build_stop_words()

    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words=stop_words,
        min_df=2, max_df=0.8,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    print(f"  TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

    # k sweep for manual inspection
    print("\n  k sweep (reconstruction error + top keywords):")
    for k in range(5, 9):
        nmf_k = NMF(n_components=k, random_state=RANDOM_SEED, max_iter=500)
        W_k = nmf_k.fit_transform(tfidf_matrix)
        print(f"\n  k={k}  err={nmf_k.reconstruction_err_:.4f}")
        for ti in range(k):
            top_w = [feature_names[i] for i in nmf_k.components_[ti].argsort()[-4:][::-1]]
            n = (W_k.argmax(axis=1) == ti).sum()
            print(f"    t{ti}: {', '.join(top_w):<40} ({n} pkgs)")

    # Fit selected k
    print(f"\n  Fitting selected k={N_RISK_TOPICS}:")
    nmf = NMF(n_components=N_RISK_TOPICS, random_state=RANDOM_SEED, max_iter=500)
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
    print(f"  Iterations: {nmf.n_iter_}")

    risk_topic_ids = W.argmax(axis=1)

    risk_topic_labels = {}
    print(f"\n  Discovered {N_RISK_TOPICS} risk topics:")
    for topic_idx in range(N_RISK_TOPICS):
        top_word_idx = H[topic_idx].argsort()[-8:][::-1]
        top_words = [feature_names[i] for i in top_word_idx]
        label = ", ".join(top_words[:4])
        risk_topic_labels[topic_idx] = label
        n_pkgs = (risk_topic_ids == topic_idx).sum()
        print(f"    Risk Topic {topic_idx}: [{label}] — {n_pkgs} pkgs")

    # Apply human-readable labels (overrides auto-labels)
    for tid, human_label in HUMAN_RISK_TOPIC_LABELS.items():
        if tid in risk_topic_labels:
            risk_topic_labels[tid] = human_label

    # Assign topic columns — NaN for non-risk rows
    df["topic_id_risk"] = np.nan
    df["topic_label_risk"] = np.nan
    risk_indices = risk_df.index
    df.loc[risk_indices, "topic_id_risk"] = risk_topic_ids
    df.loc[risk_indices, "topic_label_risk"] = [risk_topic_labels[t] for t in risk_topic_ids]

    print(f"\n  Risk topic assignment complete.")
    print(f"  Inspect auto-keywords above and update HUMAN_RISK_TOPIC_LABELS "
          f"at the top of analysis.py, then re-run.")

    return df, risk_topic_labels
```

- [ ] **Step 2: Verify `topic_model_risk()` runs and produces columns**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
risk = df[df['in_risk_universe']]
print('risk rows with topic_id_risk set:', risk['topic_id_risk'].notna().sum())
print('unique risk topics:', risk['topic_id_risk'].nunique())
assert risk['topic_id_risk'].notna().all(), 'missing topic_id_risk in risk universe'
assert df[~df['in_risk_universe']]['topic_id_risk'].isna().all(), 'non-risk rows should be NaN'
print('PASS')
"
```

Expected output includes the k sweep table and ends with `PASS`.

- [ ] **Step 3: Update HUMAN_RISK_TOPIC_LABELS with real labels**

After running Step 2, inspect the printed auto-keywords for k=6. Update `HUMAN_RISK_TOPIC_LABELS` in `analysis.py` at the top of the file with descriptive labels based on what you see. Example (will differ from actual output):

```python
HUMAN_RISK_TOPIC_LABELS = {
    0: "Legacy Java / Maven infrastructure",
    1: "npm build & task tooling",
    2: "Microsoft / .NET metapackages",
    3: "HTTP & networking utilities",
    4: "Testing & mocking",
    5: "Cloud API clients",
}
```

- [ ] **Step 4: Commit**

```bash
git add analysis.py
git commit -m "feat: add Stage 3c risk-only NMF (topic_model_risk)"
```

---

## Task 3: Rewrite `analyse_clusters()` to use risk labels

**Files:**
- Modify: `analysis.py:428-465` (`analyse_clusters()`)

- [ ] **Step 1: Rewrite `analyse_clusters()`**

Replace the entire `analyse_clusters()` function (lines 428–465) with:

```python
# ── Stage 4: Fragility analysis by cluster ─────────────────────────────

def analyse_clusters(df, risk_topic_labels):
    """Analyse fragility and criticality variation across risk-only topic clusters."""
    print("\n" + "=" * 70)
    print("STAGE 4: Fragility analysis by risk-only functional cluster")
    print("=" * 70)

    risk_df = df[df["in_risk_universe"]].copy()

    summary_rows = []
    for tid in sorted(risk_topic_labels.keys()):
        cluster = risk_df[risk_df["topic_id_risk"] == tid]
        summary_rows.append({
            "topic_id": tid,
            "label": risk_topic_labels[tid],
            "n_packages": len(cluster),
            "mean_fragility": cluster["fragility"].mean(),
            "median_fragility": cluster["fragility"].median(),
            "mean_criticality": cluster["criticality"].mean(),
            "pct_bus_factor_1": (cluster["contributors"] < 2).mean() * 100,
            "mean_days_since_release": cluster["days_since_release"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mean_fragility", ascending=False)

    print("\n  Risk cluster summary (sorted by mean fragility):\n")
    print(f"  {'Topic':<6} {'Label':<40} {'Pkgs':>5} "
          f"{'F̄':>6} {'C̄':>6} {'Bus1%':>6}")
    print("  " + "-" * 80)
    for _, row in summary.iterrows():
        print(f"  {int(row['topic_id']):<6} {row['label']:<40} {row['n_packages']:>5} "
              f"{row['mean_fragility']:>6.3f} {row['mean_criticality']:>6.3f} "
              f"{row['pct_bus_factor_1']:>5.1f}%")

    return summary
```

- [ ] **Step 2: Verify `analyse_clusters()` runs without error**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
summary = analysis.analyse_clusters(df, risk_topic_labels)
print('summary shape:', summary.shape)
assert len(summary) == analysis.N_RISK_TOPICS, f'expected {analysis.N_RISK_TOPICS} rows, got {len(summary)}'
assert summary['n_packages'].sum() == df['in_risk_universe'].sum(), 'package count mismatch'
print('PASS')
"
```

Expected: summary has N_RISK_TOPICS rows, total packages sums to 247.

- [ ] **Step 3: Commit**

```bash
git add analysis.py
git commit -m "refactor: rewrite analyse_clusters to use risk-only NMF labels"
```

---

## Task 4: Update `inspect_top_cluster()` and `compute_funding_gap()`

**Files:**
- Modify: `analysis.py:470-505` (`inspect_top_cluster()`)
- Modify: `analysis.py:678-717` (`compute_funding_gap()`)

- [ ] **Step 1: Rewrite `inspect_top_cluster()`**

Replace the entire `inspect_top_cluster()` function (lines 470–505) with:

```python
# ── Stage 4b: Inspect the largest / most fragile cluster ───────────────

def inspect_top_cluster(df, summary, risk_topic_labels):
    """Print the most notable packages in the highest-fragility risk cluster."""
    print("\n" + "=" * 70)
    print("STAGE 4b: Inspecting the most fragile risk cluster")
    print("=" * 70)

    top_tid = int(summary.iloc[0]["topic_id"])
    top_label = summary.iloc[0]["label"]
    cluster = df[(df["in_risk_universe"]) & (df["topic_id_risk"] == top_tid)].copy()

    print(f"\n  Risk Topic {top_tid} [{top_label}]")
    print(f"  {len(cluster)} packages (all in risk universe by construction)")
    print(f"  Mean fragility: {cluster['fragility'].mean():.3f}, "
          f"bus factor=1: {(cluster['contributors'] < 2).mean()*100:.1f}%")

    print(f"\n  Top 15 packages in this cluster by criticality:")
    top = cluster.nlargest(min(15, len(cluster)), "criticality")[
        ["Name", "Platform", "criticality", "fragility",
         "contributors", "dep_pkg_count"]]
    for _, row in top.iterrows():
        print(f"    {row['Name']:<40} [{row['Platform']:<10}] "
              f"C={row['criticality']:.3f} F={row['fragility']:.3f} "
              f"contrib={int(row['contributors']):>4} "
              f"deps={int(row['dep_pkg_count']):>6}")

    print(f"\n  Top 15 risk-universe packages by criticality (all clusters):")
    all_risk = df[df["in_risk_universe"]].nlargest(15, "criticality")
    for _, row in all_risk.iterrows():
        risk_label = str(row["topic_label_risk"]).split(",")[0].strip()
        print(f"    {row['Name']:<40} [{row['Platform']:<10}] "
              f"C={row['criticality']:.3f} F={row['fragility']:.3f} "
              f"cluster={risk_label}")
```

- [ ] **Step 2: Rewrite `compute_funding_gap()`**

Replace the entire `compute_funding_gap()` function (lines 678–717) with:

```python
# ── Stage 6: Funding gap by cluster ────────────────────────────────────

def compute_funding_gap(df, summary, total_gva):
    """Compute the maintenance funding gap per risk-only topic cluster."""
    print("\n" + "=" * 70)
    print("STAGE 6: Funding gap by risk-only functional cluster")
    print("=" * 70)

    cost_per_pkg = FTES_PER_PACKAGE * EU_MEDIAN_DEV_SALARY
    total_risk = int(df["in_risk_universe"].sum())
    total_gap = total_risk * cost_per_pkg

    print(f"  Risk universe: {total_risk} packages")
    print(f"  Cost per package: {FTES_PER_PACKAGE} FTE × EUR {EU_MEDIAN_DEV_SALARY:,} "
          f"= EUR {cost_per_pkg:,}/year")
    print(f"  Total funding gap: EUR {total_gap/1e6:.1f}M")
    print(f"  EU software-sector GVA: EUR {total_gva:,.0f}M")
    print(f"  Gap as fraction of GVA: {total_gap / (total_gva * 1e6) * 100:.4f}%")

    gap_rows = []
    for _, row in summary.iterrows():
        n_risk = int(row["n_packages"])
        cluster_gap = n_risk * cost_per_pkg
        gap_rows.append({
            "topic_id": row["topic_id"],
            "label": row["label"],
            "n_risk": n_risk,
            "gap_eur": cluster_gap,
            "gap_meur": cluster_gap / 1e6,
        })

    gap_df = pd.DataFrame(gap_rows).sort_values("gap_eur", ascending=False)

    print("\n  Funding gap by risk cluster:")
    for _, row in gap_df.iterrows():
        print(f"    Risk Topic {int(row['topic_id'])} [{row['label']}]: "
              f"{row['n_risk']} pkgs, EUR {row['gap_meur']:.1f}M")

    return gap_df, total_gap
```

- [ ] **Step 3: Verify both functions run**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
summary = analysis.analyse_clusters(df, risk_topic_labels)
analysis.inspect_top_cluster(df, summary, risk_topic_labels)
gva, total_gva = analysis.fetch_eurostat_gva()
gap_df, total_gap = analysis.compute_funding_gap(df, summary, total_gva)
assert abs(total_gap - len(df[df['in_risk_universe']]) * 2 * 45000) < 1, 'gap arithmetic wrong'
print('gap_df columns:', list(gap_df.columns))
print('PASS')
"
```

Expected: gap arithmetic passes, `gap_df` has columns `topic_id, label, n_risk, gap_eur, gap_meur`.

- [ ] **Step 4: Commit**

```bash
git add analysis.py
git commit -m "refactor: update inspect_top_cluster and compute_funding_gap for risk-only clusters"
```

---

## Task 5: Update `threshold_sensitivity()` — add corpus-labels note

**Files:**
- Modify: `analysis.py:599-650` (`threshold_sensitivity()`)

- [ ] **Step 1: Add corpus-labels note to `threshold_sensitivity()`**

In `threshold_sensitivity()`, find the loop over `thresholds` (around line 608). The function already uses `topic_labels` (corpus labels) and `df["topic_id"]`. Update the two references to `df["topic_id"]` inside the function to `df["topic_id_corpus"]`:

```python
# find and change (lines ~615-616):
            cluster = df[df["topic_id"] == tid]
            n_risk = len(risk[risk["topic_id"] == tid])
# to:
            cluster = df[df["topic_id_corpus"] == tid]
            n_risk = len(risk[risk["topic_id_corpus"] == tid])
```

Then add a note at the bottom of the function, just before `return results_df`:

```python
    print(f"\n  NOTE: Cluster labels above are full-corpus NMF (k={N_TOPICS}). "
          f"Risk-only clusters (Stage 3c) are not re-fit per threshold — "
          f"doing so would produce incomparable labels across thresholds.")
```

- [ ] **Step 2: Verify `threshold_sensitivity()` runs**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
summary = analysis.analyse_clusters(df, risk_topic_labels)
results = analysis.threshold_sensitivity(df, topic_labels)
assert len(results) == 3, 'expected 3 threshold rows'
print('PASS')
"
```

Expected: runs without error, prints the NOTE line at the end of Stage 4d output.

- [ ] **Step 3: Commit**

```bash
git add analysis.py
git commit -m "fix: update threshold_sensitivity to use topic_id_corpus; add corpus-labels note"
```

---

## Task 6: Update Figure 1 — grey background, risk points recoloured by risk labels

**Files:**
- Modify: `analysis.py:776-823` (Figure 1 block inside `make_figures()`)
- Modify: `analysis.py:776` — update `make_figures()` signature

- [ ] **Step 1: Update `make_figures()` signature**

Change the function signature from:

```python
def make_figures(df, summary, gap_df, boot_df, topic_labels):
```

to:

```python
def make_figures(df, summary, gap_df, boot_df, topic_labels, risk_topic_labels):
```

- [ ] **Step 2: Replace Figure 1 block**

Replace the entire Figure 1 block (lines 782–823, from `# ── Figure 1` to `plt.close(fig)` after `fig1_scatter.png`) with:

```python
    # ── Figure 1: C–F scatter — grey background, risk points by risk cluster ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background: all 9,461 packages in grey (no cluster colouring)
    ax.scatter(df["criticality"], df["fragility"],
               c="lightgrey", alpha=0.12, s=6, linewidths=0, zorder=1)

    # Risk universe: coloured by risk-only cluster
    risk = df[df["in_risk_universe"]].copy()
    n_risk_topics = int(risk["topic_id_risk"].nunique())
    cmap_risk = plt.colormaps.get_cmap("tab10").resampled(max(n_risk_topics, 1))

    for tid in sorted(risk["topic_id_risk"].dropna().unique()):
        mask = risk["topic_id_risk"] == tid
        short_label = str(risk_topic_labels[int(tid)]).split(",")[0].strip()
        ax.scatter(risk.loc[mask, "criticality"], risk.loc[mask, "fragility"],
                   c=[cmap_risk(int(tid))], alpha=0.85, s=35,
                   edgecolors="black", linewidths=0.5, zorder=2,
                   label=f"{short_label} ({mask.sum()})")

    c_thresh = df["criticality"].quantile(RISK_PERCENTILE / 100)
    f_thresh = df["fragility"].quantile(RISK_PERCENTILE / 100)
    ax.axvline(c_thresh, color="grey", linestyle="--", alpha=0.5)
    ax.axhline(f_thresh, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Criticality score (min-max normalised, [0,1])", fontsize=11)
    ax.set_ylabel("Fragility score (min-max normalised, [0,1])", fontsize=11)
    ax.set_title("Criticality vs Fragility — Risk Universe by Risk Cluster", fontsize=13)
    n_risk_count = int(df["in_risk_universe"].sum())
    fig.text(0.5, -0.02,
             f"N={len(df):,} load-bearing packages (≥100 dependent projects). "
             f"Grey background = all {len(df):,} packages (no cluster colouring). "
             f"Coloured points = {n_risk_count} risk-universe packages above both "
             f"{RISK_PERCENTILE}th-percentile thresholds (C≥{c_thresh:.3f}, F≥{f_thresh:.3f}), "
             f"coloured by risk-only NMF cluster (k={N_RISK_TOPICS}).",
             ha="center", fontsize=8, style="italic", wrap=True)
    ax.legend(title="Risk cluster (risk-only NMF)", fontsize=8, title_fontsize=9,
              loc="lower left", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_scatter.png")
```

- [ ] **Step 3: Verify Figure 1 is saved**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis, os
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
summary = analysis.analyse_clusters(df, risk_topic_labels)
gva, total_gva = analysis.fetch_eurostat_gva()
gap_df, total_gap = analysis.compute_funding_gap(df, summary, total_gva)
boot_df = analysis.bootstrap_analysis(df)
analysis.make_figures(df, summary, gap_df, boot_df, topic_labels, risk_topic_labels)
assert os.path.exists('outputs/fig1_scatter.png'), 'fig1 not saved'
print('PASS')
"
```

Expected: prints `Saved fig1_scatter.png` and `PASS`. Open `outputs/fig1_scatter.png` to visually confirm grey background and coloured risk points.

- [ ] **Step 4: Commit**

```bash
git add analysis.py
git commit -m "feat: update Fig 1 — grey background, risk points coloured by risk-only clusters"
```

---

## Task 7: Update Figure 2 (panel b → funding gap) and add Figure 0b

**Files:**
- Modify: `analysis.py:825-865` (Figure 2 block inside `make_figures()`)
- Modify: `analysis.py:303-351` (`select_topic_count()`) — add Fig 0b

- [ ] **Step 1: Replace Figure 2 block**

Replace the entire Figure 2 block (lines 825–865, from `# ── Figure 2` to `print("  Saved fig2_clusters.png")`) with:

```python
    # ── Figure 2: Risk cluster fragility (a) and funding gap (b) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel a: Mean fragility by risk cluster
    s = summary.sort_values("mean_fragility", ascending=True)
    short_labels = [str(l).split(",")[0].strip() for l in s["label"]]
    n_rc = len(s)
    cmap_rc = plt.colormaps.get_cmap("tab10").resampled(max(n_rc, 1))
    colors_a = [cmap_rc(int(tid)) for tid in s["topic_id"]]

    axes[0].barh(range(n_rc), s["mean_fragility"], color=colors_a)
    axes[0].set_yticks(range(n_rc))
    axes[0].set_yticklabels(short_labels, fontsize=9)
    axes[0].set_xlabel("Mean fragility score (min-max normalised, [0,1])")
    axes[0].set_title("(a) Mean fragility by risk cluster")
    risk_mean_f = df[df["in_risk_universe"]]["fragility"].mean()
    axes[0].axvline(risk_mean_f, color="red", linestyle="--",
                    alpha=0.7, label=f"Risk-universe mean (F̄={risk_mean_f:.3f})")
    axes[0].legend(fontsize=8)

    # Panel b: Funding gap by risk cluster (replaces risk-concentration bars,
    # which are degenerate since all risk-cluster members are in the risk universe)
    gap_sorted = gap_df.sort_values("gap_meur", ascending=True)
    short_labels_b = [str(l).split(",")[0].strip() for l in gap_sorted["label"]]
    colors_b = [cmap_rc(int(tid)) for tid in gap_sorted["topic_id"]]

    axes[1].barh(range(len(gap_sorted)), gap_sorted["gap_meur"], color=colors_b)
    axes[1].set_yticks(range(len(gap_sorted)))
    axes[1].set_yticklabels(short_labels_b, fontsize=9)
    axes[1].set_xlabel("Funding gap (EUR million/year)")
    axes[1].set_title("(b) Funding gap by risk cluster")

    n_risk_total = int(df["in_risk_universe"].sum())
    total_gap_meur = n_risk_total * FTES_PER_PACKAGE * EU_MEDIAN_DEV_SALARY / 1e6
    fig.text(0.5, -0.03,
             f"N={n_risk_total} risk-universe packages assigned to {N_RISK_TOPICS} "
             f"risk-only NMF clusters. "
             f"Panel (a): mean fragility within each cluster. "
             f"Panel (b): gross funding-gap lower bound (2 FTE × EUR 45k/pkg/yr); "
             f"total = EUR {total_gap_meur:.1f}M. "
             f"Cluster labels are human-assigned after inspecting NMF auto-keywords.",
             ha="center", fontsize=8, style="italic", wrap=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_clusters.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_clusters.png")
```

- [ ] **Step 2: Add Figure 0b inside `select_topic_count()`**

`select_topic_count()` currently saves `fig0_topic_selection.png` for the full-corpus sweep. Add a parallel risk-sweep figure by inserting the following block at the end of `select_topic_count()`, just before `return tfidf_matrix, tfidf` (around line 351):

```python
    # ── Figure 0b: risk-universe k sweep placeholder ──
    # This is a quick preview only — the definitive risk sweep runs in Stage 3c.
    # We plot k=5..8 reconstruction error for the FULL corpus here as context;
    # the actual risk-set sweep is printed (not plotted) by topic_model_risk().
    # A separate fig0b is generated there after tfidf is re-fitted on the risk set.
```

Actually, it is cleaner to generate Fig 0b inside `topic_model_risk()` since that's where the risk TF-IDF and risk sweep live. Instead, add the following block to `topic_model_risk()`, after the k sweep loop (after the line that prints the table for each k), saving `fig0b_risk_topic_selection.png`:

In `topic_model_risk()`, after the `for k in range(5, 9):` loop, add:

```python
    # Save risk k-sweep figure
    k_range_risk = list(range(5, 9))
    risk_errors = []
    for k in k_range_risk:
        nmf_k = NMF(n_components=k, random_state=RANDOM_SEED, max_iter=500)
        nmf_k.fit_transform(tfidf_matrix)
        risk_errors.append(nmf_k.reconstruction_err_)

    fig_kb, ax_kb = plt.subplots(figsize=(6, 3))
    ax_kb.plot(k_range_risk, risk_errors, "o-", color="steelblue", linewidth=2, markersize=6)
    ax_kb.axvline(N_RISK_TOPICS, color="crimson", linestyle="--", alpha=0.7,
                  label=f"Selected k={N_RISK_TOPICS}")
    ax_kb.set_xlabel("Number of topics (k)")
    ax_kb.set_ylabel("Reconstruction error (Frobenius norm)")
    ax_kb.set_title(f"Risk-universe NMF k selection (N={len(risk_df)} packages)")
    ax_kb.legend()
    ax_kb.grid(True, alpha=0.3)
    fig_kb.tight_layout()
    fig_kb.savefig(os.path.join(OUTPUT_DIR, "fig0b_risk_topic_selection.png"),
                   dpi=150, bbox_inches="tight")
    plt.close(fig_kb)
    print(f"  Saved fig0b_risk_topic_selection.png")
```

Note: this runs the NMF sweep a second time for the plot. That is acceptable (k=5..8 on 247 docs is fast — under 1 second).

- [ ] **Step 3: Verify both new figures are saved**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import analysis, os
df = analysis.load_data()
df = analysis.score_packages(df)
tfidf_matrix, tfidf = analysis.select_topic_count(df)
df, topic_labels = analysis.topic_model(df, tfidf_matrix, tfidf)
df, risk_topic_labels = analysis.topic_model_risk(df)
summary = analysis.analyse_clusters(df, risk_topic_labels)
gva, total_gva = analysis.fetch_eurostat_gva()
gap_df, total_gap = analysis.compute_funding_gap(df, summary, total_gva)
boot_df = analysis.bootstrap_analysis(df)
analysis.make_figures(df, summary, gap_df, boot_df, topic_labels, risk_topic_labels)
assert os.path.exists('outputs/fig0b_risk_topic_selection.png'), 'fig0b missing'
assert os.path.exists('outputs/fig2_clusters.png'), 'fig2 missing'
print('PASS')
"
```

Expected: both files exist. Open them to confirm Fig 2 panel (b) shows EUR funding gap bars.

- [ ] **Step 4: Commit**

```bash
git add analysis.py
git commit -m "feat: rebuild Fig 2 with funding-gap panel; add Fig 0b risk k-sweep"
```

---

## Task 8: Update `save_outputs()` and `main()` call order

**Files:**
- Modify: `analysis.py:903-929` (`save_outputs()`)
- Modify: `analysis.py:934-956` (`main()`)

- [ ] **Step 1: Update `save_outputs()`**

Replace the `out_cols` list and the `df[out_cols].to_csv(...)` section (lines 910–919) with:

```python
    out_cols = [
        "ID", "Platform", "Name", "Description",
        "dep_pkg_count", "dep_repo_count", "contributors",
        "open_issues", "stars", "days_since_release",
        "criticality", "fragility", "in_risk_universe",
        "topic_id_corpus", "topic_label_corpus",
        "topic_id_risk", "topic_label_risk",
    ]
    out_path = os.path.join(OUTPUT_DIR, "packages_scored.csv")
    df[out_cols].to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df)} rows)")
```

Also update the cluster summary save to write `risk_cluster_summary.csv` alongside `cluster_summary.csv`. After the existing `sum_path` block, add:

```python
    # Risk cluster summary (primary findings)
    risk_sum_path = os.path.join(OUTPUT_DIR, "risk_cluster_summary.csv")
    summary.to_csv(risk_sum_path, index=False)
    print(f"  Saved {risk_sum_path}")
```

(Keep the existing `cluster_summary.csv` save for corpus-level descriptive context.)

- [ ] **Step 2: Update `main()`**

Replace the entire `main()` function with:

```python
def main():
    df = load_data()
    df = score_packages(df)
    check_cf_correlation(df)
    tfidf_matrix, tfidf = select_topic_count(df)
    df, topic_labels = topic_model(df, tfidf_matrix, tfidf)
    df, risk_topic_labels = topic_model_risk(df)
    summary = analyse_clusters(df, risk_topic_labels)
    inspect_top_cluster(df, summary, risk_topic_labels)
    lasso_coefs = lasso_validation(df)
    threshold_results = threshold_sensitivity(df, topic_labels)
    gva, total_gva = fetch_eurostat_gva()
    gap_df, total_gap = compute_funding_gap(df, summary, total_gva)
    boot_df = bootstrap_analysis(df)
    make_figures(df, summary, gap_df, boot_df, topic_labels, risk_topic_labels)
    save_outputs(df, summary, gap_df)

    print("\n" + "=" * 70)
    print("DONE — All results reproduced.")
    print("=" * 70)
```

- [ ] **Step 3: Run the full pipeline end-to-end**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python analysis.py 2>&1 | tail -20
```

Expected: ends with `DONE — All results reproduced.` and no tracebacks.

- [ ] **Step 4: Verify output files**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import pandas as pd
scored = pd.read_csv('outputs/packages_scored.csv')
print('scored columns:', list(scored.columns))
assert 'topic_id_risk' in scored.columns
assert 'topic_id_corpus' in scored.columns
assert 'topic_id' not in scored.columns

risk_summary = pd.read_csv('outputs/risk_cluster_summary.csv')
print('risk_cluster_summary rows:', len(risk_summary))

gap = pd.read_csv('outputs/funding_gap_by_cluster.csv')
print('funding_gap rows:', len(gap))
total = gap['gap_meur'].sum()
print(f'total funding gap: EUR {total:.1f}M')
assert abs(total - 247 * 2 * 45000 / 1e6) < 0.1, f'gap total wrong: {total}'
print('PASS')
"
```

Expected: column assertions pass, total funding gap ≈ EUR 22.2M, `PASS`.

- [ ] **Step 5: Commit**

```bash
git add analysis.py outputs/
git commit -m "feat: update save_outputs and main — risk_cluster_summary.csv, updated column names"
```

---

## Task 9: Mirror changes in `analysis.ipynb`

**Files:**
- Modify: `analysis.ipynb`

- [ ] **Step 1: Identify which cells map to which stages**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import json
with open('analysis.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'STAGE' in src or 'def ' in src:
        preview = src[:80].replace('\n',' ')
        print(f'cell {i} ({cell[\"cell_type\"]}): {preview}')
"
```

- [ ] **Step 2: Apply all changes from Tasks 1–8 to the notebook cells**

For each change made in `analysis.py`, find the corresponding notebook cell (by matching function names or stage headers) and apply the identical change. The notebook cells should mirror `analysis.py` exactly for all modified functions:

- Constants block: add `N_RISK_TOPICS`, `HUMAN_RISK_TOPIC_LABELS`
- `topic_model()` cell: rename `topic_id` → `topic_id_corpus`, `topic_label` → `topic_label_corpus`
- After `topic_model()` cell: insert new cell with `topic_model_risk()` function definition and call
- `analyse_clusters()` cell: replace with risk-label version
- `inspect_top_cluster()` cell: replace with risk-label version
- `threshold_sensitivity()` cell: add `topic_id_corpus` references + NOTE print
- `compute_funding_gap()` cell: replace with risk-cluster version
- `make_figures()` cell(s): apply Fig 1 and Fig 2 changes
- `save_outputs()` cell: update `out_cols`, add `risk_cluster_summary.csv` save
- Execution cell at bottom: update to call `topic_model_risk()` and pass `risk_topic_labels`

- [ ] **Step 3: Clear all outputs and re-run the notebook**

```bash
cd /Users/sarahnovotny/Github/CME/final_project_v2
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb --ExecutePreprocessor.timeout=600
```

Expected: notebook executes without errors, all cells have output.

- [ ] **Step 4: Verify notebook output matches script output**

```bash
source /Users/sarahnovotny/Github/python3.12-venv/bin/activate
python -c "
import json
with open('analysis.ipynb') as f:
    nb = json.load(f)
# Check that execution_count goes from 1 to N with no gaps (no unexecuted cells)
counts = [c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']
print('execution counts:', counts)
assert all(c is not None for c in counts), 'some code cells were not executed'
print('PASS')
"
```

- [ ] **Step 5: Commit**

```bash
git add analysis.ipynb
git commit -m "feat: mirror risk-only NMF changes in analysis.ipynb; re-execute notebook"
```

---

## Self-review checklist

**Spec coverage:**
- [x] Full-corpus NMF column rename (`topic_id_corpus`/`topic_label_corpus`) — Task 1
- [x] Stage 3c `topic_model_risk()` with k sweep 5–8 and default k=6 — Task 2
- [x] TF-IDF re-fitted on risk set, `min_df=2`, `max_features=2000` — Task 2
- [x] `HUMAN_RISK_TOPIC_LABELS` placeholder dict — Task 1
- [x] `analyse_clusters()` uses risk labels — Task 3
- [x] `inspect_top_cluster()` uses risk labels — Task 4
- [x] `compute_funding_gap()` uses risk clusters — Task 4
- [x] `threshold_sensitivity()` uses corpus labels + NOTE — Task 5
- [x] Fig 1 grey background, risk points coloured by risk labels — Task 6
- [x] Fig 2 panel (b) = funding gap by risk cluster — Task 7
- [x] Fig 0b risk k-sweep — Task 7
- [x] `packages_scored.csv` gains `topic_id_risk`/`topic_label_risk`, renames corpus cols — Task 8
- [x] `risk_cluster_summary.csv` new output — Task 8
- [x] `main()` call order updated — Task 8
- [x] Notebook mirrored — Task 9

**Type/name consistency:**
- `risk_topic_labels` is a `dict[int, str]` passed from `topic_model_risk()` to `analyse_clusters()`, `inspect_top_cluster()`, `make_figures()` — consistent throughout
- `topic_id_risk` is `float` (NaN for non-risk rows) — `sorted(...dropna().unique())` handles this correctly in Fig 1 loop
- `summary["topic_id"]` is the risk topic id (int in the dict) — consistent with gap_df join

**Potential issue — int vs float for topic_id_risk:**
`df.loc[risk_indices, "topic_id_risk"] = risk_topic_ids` assigns a numpy int array. Later `risk["topic_id_risk"] == tid` where `tid` comes from `sorted(risk_topic_labels.keys())` (Python ints) will work because pandas handles int/float comparison. But `int(tid)` casts are used in print statements for safety. No fix needed.
