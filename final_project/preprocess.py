#!/usr/bin/env python3
"""
preprocess.py — Build analysis-ready datasets from raw sources.

Reads:
  raw_data/libraries-1.6.0-2020-01-12/projects_with_repository_fields-*.csv
  Eurostat nama_10_a64 (live fetch via DBnomics on first run)

Writes:
  data/packages_loadbearing.csv   — filtered, column-renamed Libraries.io slice
                                    (target ecosystems × >=100 dependent projects)
  data/eurostat_gva_2022.csv      — country-level J62/J63 GVA (cached)

This script is the only place where the raw 25 GB Libraries.io CSV and the live
Eurostat API are touched. Once it has run, the analysis notebook / script can be
re-executed end-to-end against the small CSVs in data/ with no network access
and no dependence on the raw tarball.

Run:
  python preprocess.py                 # uses cached Eurostat if available
  python preprocess.py --refresh-gva   # forces a re-fetch of Eurostat
"""

import argparse
import csv
import os
import sys

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw_data", "libraries-1.6.0-2020-01-12")
PROJECTS_FILE = os.path.join(
    RAW_DIR, "projects_with_repository_fields-1.6.0-2020-01-12.csv")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PACKAGES_OUT = os.path.join(DATA_DIR, "packages_loadbearing.csv")
EUROSTAT_OUT = os.path.join(DATA_DIR, "eurostat_gva_2022.csv")

# ── Filter parameters (must match analysis.py) ─────────────────────────
TARGET_PLATFORMS = ["NPM", "Maven", "Pypi", "Rubygems", "NuGet", "Packagist", "Cargo"]
LOAD_BEARING_THRESHOLD = 100
EUROSTAT_YEAR = 2022
EUROSTAT_AGGREGATES = ["EU27_2020", "EU28", "EA19", "EA20", "EU15"]


def build_packages_csv():
    """Filter the raw Libraries.io snapshot down to load-bearing packages."""
    print(f"[packages] Reading {PROJECTS_FILE}")
    if not os.path.exists(PROJECTS_FILE):
        sys.exit(f"ERROR: raw input not found.\n"
                 f"See raw_data/README.md for download instructions.")

    # The CSV has 59 header columns but 60 data columns (trailing comma).
    # Read header separately and append a dummy column name so pandas doesn't
    # misalign every numeric field.
    with open(PROJECTS_FILE, "r") as f:
        header = next(csv.reader(f))
    header.append("_extra")

    usecols = [
        "ID", "Platform", "Name", "Description", "Keywords",
        "Dependent Projects Count", "Dependent Repositories Count",
        "Latest Release Publish Timestamp",
        "Repository Last pushed Timestamp",
        "Repository Contributors Count",
        "Repository Open Issues Count",
        "Repository Stars Count",
    ]

    df = pd.read_csv(PROJECTS_FILE, names=header, skiprows=1,
                     usecols=usecols, low_memory=False)
    print(f"[packages] Read {len(df):,} rows from raw CSV")

    df["Platform"] = df["Platform"].str.strip()
    df = df[df["Platform"].isin(TARGET_PLATFORMS)].copy()

    numeric_cols = [
        "Dependent Projects Count", "Dependent Repositories Count",
        "Repository Contributors Count", "Repository Open Issues Count",
        "Repository Stars Count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_before = len(df)
    df = df[df["Dependent Projects Count"] >= LOAD_BEARING_THRESHOLD].copy()
    print(f"[packages] Target ecosystems: {total_before:,} rows")
    print(f"[packages] Load-bearing (>= {LOAD_BEARING_THRESHOLD} dependents): {len(df):,} rows")

    df = df.rename(columns={
        "Dependent Projects Count": "dep_pkg_count",
        "Dependent Repositories Count": "dep_repo_count",
        "Repository Contributors Count": "contributors",
        "Repository Open Issues Count": "open_issues",
        "Repository Stars Count": "stars",
        "Latest Release Publish Timestamp": "last_release",
        "Repository Last pushed Timestamp": "last_pushed",
    })

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PACKAGES_OUT, index=False)
    size_mb = os.path.getsize(PACKAGES_OUT) / 1e6
    print(f"[packages] Wrote {PACKAGES_OUT} ({len(df):,} rows, {size_mb:.1f} MB)")


def build_eurostat_csv(refresh: bool):
    """Fetch (or load cached) Eurostat J62/J63 GVA for EUROSTAT_YEAR."""
    if os.path.exists(EUROSTAT_OUT) and not refresh:
        print(f"[eurostat] Cache exists at {EUROSTAT_OUT} — skipping fetch "
              f"(use --refresh-gva to re-download)")
        return

    print(f"[eurostat] Fetching nama_10_a64 (J62_J63, B1G, CP_MEUR) via DBnomics")
    try:
        from dbnomics import fetch_series
    except ImportError:
        sys.exit("ERROR: dbnomics not installed. Run: pip install -r requirements.txt")

    series = fetch_series(
        provider_code="Eurostat",
        dataset_code="nama_10_a64",
        dimensions={"na_item": ["B1G"], "unit": ["CP_MEUR"], "nace_r2": ["J62_J63"]},
        max_nb_series=200,
    )

    series["year"] = series["period"].dt.year
    gva = (series[series["year"] == EUROSTAT_YEAR][["geo", "value"]]
           .dropna()
           .rename(columns={"geo": "country", "value": "gva_meur"}))
    gva = gva[~gva["country"].isin(EUROSTAT_AGGREGATES)].reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    gva.to_csv(EUROSTAT_OUT, index=False)
    print(f"[eurostat] Wrote {EUROSTAT_OUT} ({len(gva)} countries, "
          f"total = EUR {gva['gva_meur'].sum():,.0f}M)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--refresh-gva", action="store_true",
                    help="Force re-fetch of Eurostat data even if cached.")
    ap.add_argument("--skip-packages", action="store_true",
                    help="Skip Libraries.io rebuild (useful when only refreshing GVA).")
    args = ap.parse_args()

    if not args.skip_packages:
        build_packages_csv()
    build_eurostat_csv(refresh=args.refresh_gva)
    print("[done] Preprocessing complete.")


if __name__ == "__main__":
    main()
