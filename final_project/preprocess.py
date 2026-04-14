#!/usr/bin/env python3
"""
preprocess.py — Build analysis-ready datasets from staged raw sources.

Reads:
  raw_data/libraries-1.6.0-2020-01-12/projects_with_repository_fields-*.csv
    (staged by fetch_data.py from the Zenodo snapshot)

Writes:
  data/packages_loadbearing.csv — filtered, column-renamed Libraries.io slice
                                  (target ecosystems × >=100 dependent projects)

This script is the only place where the raw 25 GB Libraries.io CSV is touched.
Once it has run, analysis.py can be re-executed end-to-end against the small
CSVs in data/ with no network access and no dependence on the raw tarball.

The Eurostat GVA cache (data/eurostat_gva_2022.csv) is produced separately by
fetch_data.py — preprocess.py is pure-local and does no network I/O.

Run:
  python preprocess.py
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

# ── Filter parameters (must match analysis.py) ─────────────────────────
TARGET_PLATFORMS = ["NPM", "Maven", "Pypi", "Rubygems", "NuGet", "Packagist", "Cargo"]
LOAD_BEARING_THRESHOLD = 100


def build_packages_csv():
    """Filter the raw Libraries.io snapshot down to load-bearing packages."""
    print(f"[packages] Reading {PROJECTS_FILE}")
    if not os.path.exists(PROJECTS_FILE):
        sys.exit(f"ERROR: raw input not found at {PROJECTS_FILE}.\n"
                 f"Run `python fetch_data.py` first (see raw_data/README.md).")

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()
    build_packages_csv()
    print("[done] Preprocessing complete.")


if __name__ == "__main__":
    main()
