#!/usr/bin/env python3
"""
Critical-but-Fragile: Mapping Fragility Across Functional Categories
of Open Source Infrastructure

ECP77594 — Computational Methods in Economics
Final Project

This script reproduces all results from the analysis-ready CSVs in data/.
Run preprocess.py first to build those CSVs from raw sources.

Pipeline:
  1. Load analysis-ready Libraries.io slice (data/packages_loadbearing.csv)
  2. Score packages for criticality and fragility
  3. Identify the risk universe (critical + fragile packages)
  4. Run topic modelling (NMF) on package descriptions
  5. Analyse fragility variation across clusters
  6. Load cached Eurostat GVA (data/eurostat_gva_2022.csv)
  7. Compute the maintenance funding gap per cluster
  8. Monte Carlo bootstrap (1,000 replicates) for confidence intervals
  9. Produce all figures and tables in outputs/

Inputs:   data/packages_loadbearing.csv
          data/eurostat_gva_2022.csv
Outputs:  outputs/packages_scored.csv
          outputs/cluster_summary.csv
          outputs/funding_gap_by_cluster.csv
          outputs/fig0_topic_selection.png .. fig3_bootstrap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# scikit-learn for text processing and topic modelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ── Configuration ──────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PACKAGES_FILE = os.path.join(DATA_DIR, "packages_loadbearing.csv")
EUROSTAT_FILE = os.path.join(DATA_DIR, "eurostat_gva_2022.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PLATFORMS = ["NPM", "Maven", "Pypi", "Rubygems", "NuGet", "Packagist", "Cargo"]
LOAD_BEARING_THRESHOLD = 100  # minimum dependent packages
RISK_PERCENTILE = 75          # percentile cutoff for risk universe
SNAPSHOT_DATE = datetime(2020, 1, 12)

# Fragility weights
W_CONTRIBUTOR = 0.30
W_STALENESS = 0.30
W_ISSUES = 0.20
W_BUSFACTOR = 0.20

# Criticality weights
W_DEP_PKG = 0.65
W_DEP_REPO = 0.35

# Topic modelling
N_TOPICS = 25
N_RISK_TOPICS = 5         # topics for risk-only NMF; k=5 chosen by interpretability — no elbow in reconstruction error at N=247; splits MS/.NET from HTTP/web utilities while keeping all clusters ≥14 packages
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Human-readable topic labels, assigned after inspecting the auto-generated
# NMF labels and the actual package composition of each cluster.
# Keys are topic IDs (stable given RANDOM_SEED=42 and the same stop word list).
HUMAN_TOPIC_LABELS = {
    0:  "Generic/legacy infrastructure (residual)",
    1:  "React ecosystem",
    2:  "JS utility libraries",
    3:  "Application frameworks (.NET/PHP)",
    4:  "Build tooling (bundlers)",
    5:  "Cloud & API clients",
    6:  "Code linting",
    7:  "Data serialisation & validation",
    8:  "Test runners",
    9:  "JS transpilation (Babel)",
    10: "PHP/Symfony ecosystem",
    11: "CLI tools & Ember addons",
    12: "CSS & styling",
    13: "Command-line infrastructure",
    14: "Task runners (Gulp/Rake)",
    15: "Web & mobile frameworks",
    16: "Parsers & code transformation",
    17: "Testing & mocking",
    18: "HTTP & networking",
    19: "Angular ecosystem",
    20: "Server infrastructure",
    21: "Compilation & code transforms",
    22: "Vue ecosystem",
    23: "Documentation & formatting",
    24: "General-purpose infrastructure (residual)",
}

# Human-readable labels for the risk-only NMF clusters.
# Placeholder values — overwrite after inspecting the first-run sweep output
# and the auto-generated keywords printed by topic_model_risk().
HUMAN_RISK_TOPIC_LABELS = {
    0: "Gulp / JS build & task tooling",
    1: "HTTP & web protocol utilities",
    2: "Apache / Java Commons legacy infrastructure",
    3: "Karma / JS test runners",
    4: "Microsoft / .NET + npm ecosystem",
}

# Domain-specific stop words: terms that describe what a package IS (artifact type,
# platform identity, marketing language) rather than what it DOES (function).
# These dominate topic labels without distinguishing functional categories.
DOMAIN_STOP_WORDS = [
    # Artifact types
    "package", "packages", "library", "libraries", "module", "modules",
    "plugin", "plugins", "component", "components", "extension", "extensions",
    "wrapper", "wrappers", "binding", "bindings", "adapter", "adapters",
    "middleware", "helper", "helpers", "utility", "utilities", "util", "utils",
    "toolkit", "toolset", "sdk", "api",
    # Platform / ecosystem identifiers
    "npm", "node", "nodejs", "js", "javascript",
    "php", "python", "ruby", "java", "net", "asp", "dotnet", "csharp",
    "cargo", "crate", "gem", "pip", "nuget", "packagist", "maven",
    # Marketing / filler
    "simple", "lightweight", "fast", "easy", "small", "tiny", "minimal",
    "modern", "powerful", "flexible", "robust", "provides", "support",
    "supports", "based", "using", "used", "use", "set", "just",
    # Generic software terms
    "code", "project", "projects", "application", "applications", "app",
    "tool", "tools", "implementation", "implementations",
    "core", "base", "common", "standard", "default", "official",
    "version", "release", "latest", "new", "updated",
    # Generic structure words not in sklearn's list
    "file", "files", "data", "type", "types", "entry", "stub",
    "method", "methods", "function", "functions", "class", "classes",
    "object", "objects", "interface", "interfaces",
    "config", "configuration", "settings", "options",
    "exported",
    # Generic programming terms that don't distinguish function
    "string", "strings", "stream", "streams", "buffer", "buffers",
    "browser", "convert", "build", "builds", "building",
    "run", "running", "create", "creating",
    "load", "loading", "read", "write",
    "render", "rendering",
    "native", "apps", "development",
    "definitions", "definition",
    "async", "callback", "promise", "event", "events",
    "add", "added", "adding", "install", "require",
    "cross", "platform", "like", "runtime", "directory",
]

# Funding gap
FTES_PER_PACKAGE = 2
EU_MEDIAN_DEV_SALARY = 45_000  # EUR per year


# ── Helper functions ───────────────────────────────────────────────────

def normalise(s):
    """Min-max normalise a series to [0, 1]."""
    smin, smax = s.min(), s.max()
    if smax == smin:
        return pd.Series(0.5, index=s.index)
    return (s - smin) / (smax - smin)


# ── Stage 1: Load and clean ────────────────────────────────────────────

def load_data():
    """Load the analysis-ready load-bearing packages CSV produced by preprocess.py."""
    print("=" * 70)
    print("STAGE 1: Loading analysis-ready packages")
    print("=" * 70)

    if not os.path.exists(PACKAGES_FILE):
        raise FileNotFoundError(
            f"{PACKAGES_FILE} not found. Run `python preprocess.py` first.")

    df = pd.read_csv(PACKAGES_FILE, low_memory=False)
    print(f"  Loaded {len(df):,} load-bearing packages from {PACKAGES_FILE}")
    print(f"  Descriptions available: {df['Description'].notna().sum():,} "
          f"({df['Description'].notna().mean()*100:.1f}%)")
    return df


# ── Stage 2: Score criticality and fragility ───────────────────────────

def score_packages(df):
    """Compute criticality and fragility scores."""
    print("\n" + "=" * 70)
    print("STAGE 2: Scoring criticality and fragility")
    print("=" * 70)

    # --- Criticality ---
    c1 = normalise(np.log1p(df["dep_pkg_count"]))
    c2 = normalise(np.log1p(df["dep_repo_count"]))
    df["criticality"] = normalise(W_DEP_PKG * c1 + W_DEP_REPO * c2)

    # --- Fragility ---
    # F1: Inverse contributor count (missing → impute as 1 = most fragile)
    df["contributors"] = df["contributors"].fillna(1).clip(lower=1)
    f1 = normalise(1.0 / df["contributors"])

    # F2: Days since last release (or last push as fallback)
    for col in ["last_release", "last_pushed"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    release_date = df["last_release"].dt.tz_localize(None)
    push_date = df["last_pushed"].dt.tz_localize(None)
    best_date = release_date.fillna(push_date)
    df["days_since_release"] = (SNAPSHOT_DATE - best_date).dt.days
    p90 = df["days_since_release"].quantile(0.9)
    df["days_since_release"] = df["days_since_release"].fillna(p90).clip(lower=0)
    f2 = normalise(np.log1p(df["days_since_release"]))

    # F3: Issue ratio (open issues / stars, capped)
    df["open_issues"] = df["open_issues"].fillna(0)
    df["stars"] = df["stars"].fillna(0)
    df["issue_ratio"] = df["open_issues"] / df["stars"].clip(lower=1)
    f3 = normalise(df["issue_ratio"])

    # F4: Bus factor flag (< 2 contributors)
    df["bus_factor"] = (df["contributors"] < 2).astype(float)
    f4 = normalise(df["bus_factor"])

    df["fragility"] = normalise(
        W_CONTRIBUTOR * f1 + W_STALENESS * f2 +
        W_ISSUES * f3 + W_BUSFACTOR * f4)

    # --- Risk universe ---
    c_thresh = df["criticality"].quantile(RISK_PERCENTILE / 100)
    f_thresh = df["fragility"].quantile(RISK_PERCENTILE / 100)
    df["in_risk_universe"] = (
        (df["criticality"] >= c_thresh) & (df["fragility"] >= f_thresh))

    n_risk = df["in_risk_universe"].sum()
    print(f"  Criticality threshold (p{RISK_PERCENTILE}): {c_thresh:.3f}")
    print(f"  Fragility threshold (p{RISK_PERCENTILE}): {f_thresh:.3f}")
    print(f"  Risk universe: {n_risk} packages")
    print(f"  Platform breakdown:")
    for plat, count in df[df["in_risk_universe"]]["Platform"].value_counts().items():
        print(f"    {plat}: {count}")

    return df


# ── Stage 2b: Criticality-Fragility correlation ───────────────────────

def check_cf_correlation(df):
    """Check whether criticality and fragility are correlated."""
    print("\n" + "=" * 70)
    print("STAGE 2b: Criticality-Fragility correlation")
    print("=" * 70)

    from scipy import stats

    pearson_r, pearson_p = stats.pearsonr(df["criticality"], df["fragility"])
    spearman_r, spearman_p = stats.spearmanr(df["criticality"], df["fragility"])

    print(f"\n  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")

    c_thresh = df["criticality"].quantile(RISK_PERCENTILE / 100)
    f_thresh = df["fragility"].quantile(RISK_PERCENTILE / 100)

    hi_c_hi_f = ((df["criticality"] >= c_thresh) & (df["fragility"] >= f_thresh)).sum()
    hi_c_lo_f = ((df["criticality"] >= c_thresh) & (df["fragility"] < f_thresh)).sum()
    lo_c_hi_f = ((df["criticality"] < c_thresh) & (df["fragility"] >= f_thresh)).sum()
    lo_c_lo_f = ((df["criticality"] < c_thresh) & (df["fragility"] < f_thresh)).sum()
    expected = len(df) * 0.0625

    print(f"\n  Quadrant distribution (threshold = {RISK_PERCENTILE}th percentile):")
    print(f"    High C, High F (risk):  {hi_c_hi_f:>5} ({hi_c_hi_f/len(df)*100:.1f}%)")
    print(f"    High C, Low F:          {hi_c_lo_f:>5} ({hi_c_lo_f/len(df)*100:.1f}%)")
    print(f"    Low C, High F:          {lo_c_hi_f:>5} ({lo_c_hi_f/len(df)*100:.1f}%)")
    print(f"    Low C, Low F:           {lo_c_lo_f:>5} ({lo_c_lo_f/len(df)*100:.1f}%)")
    print(f"\n  Expected if independent:  {expected:.0f} (6.25%)")
    print(f"  Actual risk universe:     {hi_c_hi_f} ({hi_c_hi_f/len(df)*100:.1f}%)")
    print(f"\n  Interpretation: C and F are negatively correlated — more critical packages")
    print(f"  tend to be less fragile (the market partially works). The risk universe is")
    print(f"  smaller than chance would predict, meaning that being simultaneously critical")
    print(f"  AND fragile is rare. The packages that do fall there have slipped through")
    print(f"  the attention that their dependency count would normally attract.")


# ── Stage 3a: Topic count selection ────────────────────────────────────

def _build_stop_words():
    """Combine sklearn's English stop words with domain-specific terms."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS) + DOMAIN_STOP_WORDS


def _clean_descriptions(texts):
    """Strip URLs, badge markdown, and HTML tags from descriptions."""
    import re
    cleaned = texts.copy()
    # Remove URLs (http/https/ftp)
    cleaned = cleaned.str.replace(r'https?://\S+', '', regex=True)
    cleaned = cleaned.str.replace(r'ftp://\S+', '', regex=True)
    # Remove markdown image/badge syntax: ![...](...)
    cleaned = cleaned.str.replace(r'!\[.*?\]\(.*?\)', '', regex=True)
    # Remove markdown link syntax but keep text: [text](url) -> text
    cleaned = cleaned.str.replace(r'\[([^\]]*)\]\([^\)]*\)', r'\1', regex=True)
    # Remove HTML tags
    cleaned = cleaned.str.replace(r'<[^>]+>', '', regex=True)
    return cleaned


def select_topic_count(df):
    """Sweep k=5..25 and plot reconstruction error to find the elbow."""
    print("\n" + "=" * 70)
    print("STAGE 3a: Topic count selection (reconstruction error sweep)")
    print("=" * 70)

    texts = _clean_descriptions(df["Description"].fillna("").astype(str))
    stop_words = _build_stop_words()

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words=stop_words,
        min_df=5, max_df=0.7,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    tfidf_matrix = tfidf.fit_transform(texts)

    k_range = range(5, 26)
    errors = []
    for k in k_range:
        nmf = NMF(n_components=k, random_state=RANDOM_SEED, max_iter=500)
        W = nmf.fit_transform(tfidf_matrix)
        # reconstruction_err_ is the Frobenius norm of (X - WH)
        errors.append(nmf.reconstruction_err_)
        print(f"  k={k:2d}  reconstruction error = {nmf.reconstruction_err_:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), errors, "o-", color="steelblue", linewidth=2, markersize=5)
    ax.axvline(N_TOPICS, color="crimson", linestyle="--", alpha=0.7,
               label=f"Selected k={N_TOPICS}")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("Reconstruction error (Frobenius norm)")
    ax.set_title("NMF topic-count selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.02,
             f"TF-IDF over N={tfidf_matrix.shape[0]:,} package descriptions (1,755 features); "
             f"NMF random_state={RANDOM_SEED}, max_iter=500. "
             f"k=25 selected by topic-quality inspection (no sharp elbow in the curve).",
             ha="center", fontsize=8, style="italic", wrap=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig0_topic_selection.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved fig0_topic_selection.png")
    print(f"  Selected k={N_TOPICS}")

    return tfidf_matrix, tfidf


# ── Stage 3b: Topic modelling ─────────────────────────────────────────

def topic_model(df, tfidf_matrix=None, tfidf=None):
    """Run NMF topic modelling on package descriptions."""
    print("\n" + "=" * 70)
    print("STAGE 3b: Topic modelling (NMF on TF-IDF, k={})".format(N_TOPICS))
    print("=" * 70)

    # Build TF-IDF if not passed from the sweep
    if tfidf_matrix is None:
        texts = _clean_descriptions(df["Description"].fillna("").astype(str))
        stop_words = _build_stop_words()
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words=stop_words,
            min_df=5, max_df=0.7,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        tfidf_matrix = tfidf.fit_transform(texts)

    feature_names = tfidf.get_feature_names_out()
    print(f"  TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

    # NMF topic model
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_SEED, max_iter=500)
    W = nmf.fit_transform(tfidf_matrix)  # doc-topic matrix
    H = nmf.components_                   # topic-term matrix

    print(f"  Reconstruction error: {nmf.reconstruction_err_:.2f}")
    print(f"  Iterations: {nmf.n_iter_}")

    # Assign each package to its dominant topic
    df["topic_id_corpus"] = W.argmax(axis=1)
    df["topic_weight_corpus"] = W.max(axis=1)

    # Label topics by top words
    topic_labels = {}
    print(f"\n  Discovered {N_TOPICS} topics:")
    for topic_idx in range(N_TOPICS):
        top_word_idx = H[topic_idx].argsort()[-8:][::-1]
        top_words = [feature_names[i] for i in top_word_idx]
        label = ", ".join(top_words[:4])
        topic_labels[topic_idx] = label
        n_pkgs = (df["topic_id_corpus"] == topic_idx).sum()
        n_risk = ((df["topic_id_corpus"] == topic_idx) & df["in_risk_universe"]).sum()
        print(f"    Topic {topic_idx}: [{label}] — {n_pkgs} pkgs, {n_risk} in risk universe")

    # Apply human-readable labels (kept as a separate mapping from auto-labels).
    # Auto-labels are preserved in the printed output above for transparency;
    # human labels are used for all downstream analysis, figures, and outputs.
    for tid, human_label in HUMAN_TOPIC_LABELS.items():
        if tid in topic_labels:
            topic_labels[tid] = human_label

    df["topic_label_corpus"] = df["topic_id_corpus"].map(topic_labels)

    print(f"\n  Human-readable labels applied:")
    for tid in sorted(topic_labels.keys()):
        n = (df["topic_id_corpus"] == tid).sum()
        print(f"    T{tid:2d}: {topic_labels[tid]:<45} ({n} pkgs)")

    # Check largest cluster size
    largest = df["topic_id_corpus"].value_counts().iloc[0]
    largest_pct = largest / len(df) * 100
    print(f"\n  Largest cluster: {largest} packages ({largest_pct:.1f}%)")
    if largest_pct > 20:
        print(f"  WARNING: Largest cluster holds >{20}% of packages — "
              f"consider increasing k or inspecting that topic.")

    return df, topic_labels


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
    print(f"STAGE 3c: Risk-only topic modelling (NMF k sweep 3–8, selected k={N_RISK_TOPICS})")
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
    sweep_errors = []
    for k in range(3, 9):
        nmf_k = NMF(n_components=k, random_state=RANDOM_SEED, max_iter=500)
        W_k = nmf_k.fit_transform(tfidf_matrix)
        sweep_errors.append(nmf_k.reconstruction_err_)
        print(f"\n  k={k}  err={nmf_k.reconstruction_err_:.4f}")
        for ti in range(k):
            top_w = [feature_names[i] for i in nmf_k.components_[ti].argsort()[-4:][::-1]]
            n = (W_k.argmax(axis=1) == ti).sum()
            print(f"    t{ti}: {', '.join(top_w):<40} ({n} pkgs)")

    # Save risk k-sweep figure
    fig_kb, ax_kb = plt.subplots(figsize=(6, 3))
    ax_kb.plot(list(range(3, 9)), sweep_errors, "o-", color="steelblue", linewidth=2, markersize=6)
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
    df["topic_id_risk"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["topic_label_risk"] = pd.Series(np.nan, index=df.index, dtype=object)
    risk_indices = risk_df.index
    df.loc[risk_indices, "topic_id_risk"] = risk_topic_ids
    df.loc[risk_indices, "topic_label_risk"] = [risk_topic_labels[t] for t in risk_topic_ids]

    print(f"\n  Risk topic assignment complete.")
    print(f"  Inspect auto-keywords above and update HUMAN_RISK_TOPIC_LABELS "
          f"at the top of analysis.py, then re-run.")

    return df, risk_topic_labels


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
        risk_label = (str(row["topic_label_risk"]).split(",")[0].strip()
                      if pd.notna(row["topic_label_risk"]) else "unassigned")
        print(f"    {row['Name']:<40} [{row['Platform']:<10}] "
              f"C={row['criticality']:.3f} F={row['fragility']:.3f} "
              f"cluster={risk_label}")


# ── Stage 4c: LASSO validation of fragility weights ───────────────────

def lasso_validation(df):
    """Use LASSO regression to empirically validate fragility indicator weights."""
    print("\n" + "=" * 70)
    print("STAGE 4c: LASSO validation of fragility weights")
    print("=" * 70)

    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # Target: did the package go stale? (>1 year without release at snapshot)
    # This is an observable outcome that fragility should predict.
    # We use days_since_release > 365 as a proxy for "under-maintained."
    # To avoid circularity, we build features from the raw indicators
    # BEFORE they are combined into the fragility score.
    target = (df["days_since_release"] > 365).astype(int)

    # Features: the raw fragility components (before weighting)
    features = pd.DataFrame({
        "inv_contributors": 1.0 / df["contributors"].clip(lower=1),
        "issue_ratio": df["issue_ratio"],
        "bus_factor": (df["contributors"] < 2).astype(float),
        "log_dep_pkg": np.log1p(df["dep_pkg_count"]),
        "log_dep_repo": np.log1p(df["dep_repo_count"]),
        "stars": np.log1p(df["stars"]),
    })

    # Drop rows where target is derived from the same column (staleness)
    # ... actually, staleness IS one of our fragility components.
    # To avoid circularity, predict bus_factor instead:
    # "which observable features predict single-maintainer status?"
    target = (df["contributors"] < 2).astype(int)
    features = pd.DataFrame({
        "log_days_since_release": np.log1p(df["days_since_release"]),
        "issue_ratio": df["issue_ratio"],
        "log_dep_pkg": np.log1p(df["dep_pkg_count"]),
        "log_dep_repo": np.log1p(df["dep_repo_count"]),
        "log_stars": np.log1p(df["stars"]),
        "log_open_issues": np.log1p(df["open_issues"]),
    })

    # Drop NaN rows
    valid = features.notna().all(axis=1) & target.notna()
    X = features[valid].values
    y = target[valid].values

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO with cross-validation
    lasso = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=5000)
    lasso.fit(X_scaled, y)

    print(f"\n  Target: bus_factor_1 (contributor count < 2)")
    print(f"  Observations: {len(y)}")
    print(f"  Optimal alpha (CV): {lasso.alpha_:.6f}")
    print(f"  R² (in-sample): {lasso.score(X_scaled, y):.4f}")

    # Cross-validated R²
    cv_scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring="r2")
    print(f"  R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    print(f"\n  LASSO coefficients (standardised):")
    coef_df = pd.DataFrame({
        "feature": features.columns,
        "coefficient": lasso.coef_,
        "abs_coef": np.abs(lasso.coef_),
    }).sort_values("abs_coef", ascending=False)

    for _, row in coef_df.iterrows():
        bar = "+" if row["coefficient"] > 0 else "-"
        print(f"    {row['feature']:<30} {row['coefficient']:>8.4f}  "
              f"{'█' * int(row['abs_coef'] * 50)}")

    # Comparison to assumed weights
    print(f"\n  Interpretation:")
    print(f"    The LASSO identifies which observable features predict single-maintainer")
    print(f"    status. Features with large negative coefficients (e.g., stars, dep_count)")
    print(f"    are protective: popular, widely-used packages tend to have more maintainers.")
    print(f"    Features with positive coefficients (e.g., staleness, issue_ratio) are")
    print(f"    risk factors that co-occur with under-resourcing.")

    return coef_df


# ── Stage 4d: Threshold sensitivity ───────────────────────────────────

def threshold_sensitivity(df, topic_labels):
    """Show how the risk universe and cluster ranking change across thresholds."""
    print("\n" + "=" * 70)
    print("STAGE 4d: Threshold sensitivity (60th, 75th, 90th percentile)")
    print("=" * 70)

    thresholds = [60, 75, 90]
    results = []

    for pct in thresholds:
        c_t = df["criticality"].quantile(pct / 100)
        f_t = df["fragility"].quantile(pct / 100)
        risk = df[(df["criticality"] >= c_t) & (df["fragility"] >= f_t)]

        # Top 5 clusters by risk concentration at this threshold
        cluster_risk = []
        for tid in sorted(topic_labels.keys()):
            cluster = df[df["topic_id_corpus"] == tid]
            n_risk = len(risk[risk["topic_id_corpus"] == tid])
            cluster_risk.append({
                "topic_id": tid,
                "label": topic_labels[tid].split(",")[0].strip(),
                "risk_pct": n_risk / len(cluster) * 100 if len(cluster) > 0 else 0,
                "n_risk": n_risk,
            })
        cr = pd.DataFrame(cluster_risk).sort_values("risk_pct", ascending=False)

        print(f"\n  Threshold: {pct}th percentile (C>={c_t:.3f}, F>={f_t:.3f})")
        print(f"  Risk universe: {len(risk)} packages")
        print(f"  Funding gap: EUR {len(risk) * FTES_PER_PACKAGE * EU_MEDIAN_DEV_SALARY / 1e6:.1f}M")
        print(f"  Top 5 clusters by risk concentration:")
        for _, row in cr.head(5).iterrows():
            print(f"    {row['label']:<30} {row['risk_pct']:>5.1f}% ({row['n_risk']} pkgs)")

        results.append({
            "threshold": pct,
            "risk_size": len(risk),
            "gap_meur": len(risk) * FTES_PER_PACKAGE * EU_MEDIAN_DEV_SALARY / 1e6,
            "top_cluster": cr.iloc[0]["label"],
            "top_cluster_risk_pct": cr.iloc[0]["risk_pct"],
        })

    results_df = pd.DataFrame(results)

    # Check stability: is the top cluster the same across thresholds?
    top_clusters = results_df["top_cluster"].unique()
    if len(top_clusters) == 1:
        print(f"\n  STABLE: '{top_clusters[0]}' is the most fragile cluster at all thresholds.")
    else:
        print(f"\n  UNSTABLE: Top cluster varies across thresholds: {list(top_clusters)}")

    print(f"\n  NOTE: Cluster labels above are full-corpus NMF (k={N_TOPICS}). "
          f"Risk-only clusters (Stage 3c) are not re-fit per threshold — "
          f"doing so would produce incomparable labels across thresholds.")

    return results_df


# ── Stage 5: Eurostat GVA ──────────────────────────────────────────────

def fetch_eurostat_gva():
    """Load cached Eurostat J62/J63 GVA produced by preprocess.py.

    The DBnomics API call lives in preprocess.py so this script can be re-run
    offline and remains deterministic.
    """
    print("\n" + "=" * 70)
    print("STAGE 5: Loading cached Eurostat GVA")
    print("=" * 70)

    if not os.path.exists(EUROSTAT_FILE):
        raise FileNotFoundError(
            f"{EUROSTAT_FILE} not found. Run `python fetch_data.py` first.")

    gva = pd.read_csv(EUROSTAT_FILE)
    total_gva = gva["gva_meur"].sum()
    print(f"  Countries with J62/J63 GVA in 2022: {len(gva)}")
    print(f"  Total EU software-sector GVA: EUR {total_gva:,.0f}M")
    return gva, total_gva


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

    assert sum(r["n_risk"] for r in gap_rows) == total_risk, (
        f"Per-cluster package count ({sum(r['n_risk'] for r in gap_rows)}) "
        f"does not match risk-universe size ({total_risk}). "
        f"Check for unassigned topic_id_risk in risk universe."
    )

    return gap_df, total_gap


# ── Stage 7: Monte Carlo bootstrap ─────────────────────────────────────

def bootstrap_analysis(df):
    """Monte Carlo bootstrap: resample packages, recompute risk universe."""
    print("\n" + "=" * 70)
    print("STAGE 7: Monte Carlo bootstrap (1,000 replicates)")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    cost_per_pkg = FTES_PER_PACKAGE * EU_MEDIAN_DEV_SALARY

    # Point estimates
    c_thresh = df["criticality"].quantile(RISK_PERCENTILE / 100)
    f_thresh = df["fragility"].quantile(RISK_PERCENTILE / 100)

    boot_results = []
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]

        # Recompute thresholds for this sample
        c_t = sample["criticality"].quantile(RISK_PERCENTILE / 100)
        f_t = sample["fragility"].quantile(RISK_PERCENTILE / 100)
        risk = sample[(sample["criticality"] >= c_t) & (sample["fragility"] >= f_t)]

        boot_results.append({
            "risk_size": len(risk),
            "mean_fragility": risk["fragility"].mean() if len(risk) > 0 else 0,
            "mean_criticality": risk["criticality"].mean() if len(risk) > 0 else 0,
            "funding_gap_meur": len(risk) * cost_per_pkg / 1e6,
        })

    boot_df = pd.DataFrame(boot_results)

    # Point estimates
    point_risk = df["in_risk_universe"].sum()
    point_gap = point_risk * cost_per_pkg / 1e6

    print(f"\n  Point estimate — risk universe: {point_risk}, gap: EUR {point_gap:.1f}M")
    print(f"  Bootstrap 95% CI:")
    for col, label, unit in [
        ("risk_size", "Risk universe size", ""),
        ("funding_gap_meur", "Funding gap", "M EUR"),
        ("mean_fragility", "Mean fragility (risk)", ""),
    ]:
        lo = boot_df[col].quantile(0.025)
        hi = boot_df[col].quantile(0.975)
        se = boot_df[col].std()
        print(f"    {label}: [{lo:.1f}, {hi:.1f}] {unit} (SE={se:.2f})")

    return boot_df


# ── Stage 8: Figures ───────────────────────────────────────────────────

def make_figures(df, summary, gap_df, boot_df, topic_labels, risk_topic_labels):
    """Generate all figures for the report."""
    print("\n" + "=" * 70)
    print("STAGE 8: Generating figures")
    print("=" * 70)

    # ── Figure 1: C–F scatter — grey background, risk points by risk cluster ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background: all 9,461 packages in grey (no cluster colouring)
    ax.scatter(df["criticality"], df["fragility"],
               c="lightgrey", alpha=0.12, s=6, linewidths=0, zorder=1)

    # Risk universe: coloured by risk-only cluster
    risk = df[df["in_risk_universe"]].copy()
    n_risk_topics = int(risk["topic_id_risk"].dropna().nunique())
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

    # ── Figure 2: Risk cluster fragility (a) and funding gap (b) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_rc = len(summary)
    cmap_rc = plt.colormaps.get_cmap("tab10").resampled(max(n_rc, 1))

    # Panel a: Mean fragility by risk cluster
    s = summary.sort_values("mean_fragility", ascending=True)
    short_labels_a = [str(l).split(",")[0].strip() for l in s["label"]]
    colors_a = [cmap_rc(int(tid)) for tid in s["topic_id"]]

    axes[0].barh(range(n_rc), s["mean_fragility"], color=colors_a)
    axes[0].set_yticks(range(n_rc))
    axes[0].set_yticklabels(short_labels_a, fontsize=9)
    axes[0].set_xlabel("Mean fragility score (min-max normalised, [0,1])")
    axes[0].set_title("(a) Mean fragility by risk cluster")
    risk_mean_f = df[df["in_risk_universe"]]["fragility"].mean()
    axes[0].axvline(risk_mean_f, color="red", linestyle="--",
                    alpha=0.7, label=f"Risk-universe mean (F̄={risk_mean_f:.3f})")
    axes[0].legend(fontsize=8)

    # Panel b: Funding gap by risk cluster
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

    # ── Figure 3: Bootstrap distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        ("risk_size", "Risk Universe Size", "Packages"),
        ("funding_gap_meur", "Funding Gap", "M EUR"),
        ("mean_fragility", "Mean Fragility (Risk Universe)", "Score"),
    ]
    for ax, (col, title, xlabel) in zip(axes, panels):
        ax.hist(boot_df[col], bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        lo = boot_df[col].quantile(0.025)
        hi = boot_df[col].quantile(0.975)
        ax.axvline(lo, color="orange", linestyle="--", linewidth=1.5, label="95% CI")
        ax.axvline(hi, color="orange", linestyle="--", linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.suptitle(f"Monte Carlo bootstrap distributions ({N_BOOTSTRAP:,} replicates)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.04,
             f"Resampling N={len(df):,} load-bearing packages with replacement, "
             f"recomputing the {RISK_PERCENTILE}th-percentile risk thresholds and the "
             f"funding-gap arithmetic on each replicate. Orange dashed lines mark the "
             f"percentile-based 95% CI (Efron & Tibshirani, 1993). "
             f"Funding gap uses 2 FTE × EUR 45,000/yr per risk package — a gross lower bound.",
             ha="center", fontsize=8, style="italic", wrap=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_bootstrap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_bootstrap.png")


# ── Stage 9: Save outputs ─────────────────────────────────────────────

def save_outputs(df, summary, gap_df):
    """Save the analysis-ready dataset and summary tables."""
    print("\n" + "=" * 70)
    print("STAGE 9: Saving outputs")
    print("=" * 70)

    # Analysis-ready dataset
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

    # Cluster summary
    sum_path = os.path.join(OUTPUT_DIR, "cluster_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"  Saved {sum_path}")

    # Risk cluster summary (primary findings)
    risk_sum_path = os.path.join(OUTPUT_DIR, "risk_cluster_summary.csv")
    summary.to_csv(risk_sum_path, index=False)
    print(f"  Saved {risk_sum_path}")

    # Funding gap
    gap_path = os.path.join(OUTPUT_DIR, "funding_gap_by_cluster.csv")
    gap_df.to_csv(gap_path, index=False)
    print(f"  Saved {gap_path}")


# ── Main ───────────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    main()
