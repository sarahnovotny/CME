#!/usr/bin/env python3
"""
fetch_data.py — Download and stage raw inputs for the pipeline.

Fetches:
  1. Libraries.io v1.6.0 snapshot (Zenodo record 3626071, ~25 GB tar.gz).
     Stream-extracts ONLY projects_with_repository_fields-1.6.0-2020-01-12.csv
     into raw_data/libraries-1.6.0-2020-01-12/, skipping the ~134 GB of
     tarball members preprocess.py does not consume.
  2. Eurostat nama_10_a64 (J62_J63, B1G, CP_MEUR) for EUROSTAT_YEAR via
     DBnomics, written to data/eurostat_gva_2022.csv.

After this runs, preprocess.py builds the analysis-ready CSVs offline and
analysis.py produces all results, tables, and figures.

Run:
  python fetch_data.py                 # idempotent; skips outputs already present
  python fetch_data.py --force         # re-download + re-extract + re-fetch
  python fetch_data.py --skip-zenodo   # Eurostat only
  python fetch_data.py --skip-eurostat # Libraries.io only
  python fetch_data.py --keep-tarball  # retain the 25 GB archive after extraction
"""

import argparse
import hashlib
import os
import sys
import tarfile

import requests
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw_data")
LIBRARIES_DIR = os.path.join(RAW_DIR, "libraries-1.6.0-2020-01-12")
TARBALL_PATH = os.path.join(RAW_DIR, "libraries-1.6.0-2020-01-12.tar.gz")
TARGET_CSV_NAME = "projects_with_repository_fields-1.6.0-2020-01-12.csv"
TARGET_CSV_PATH = os.path.join(LIBRARIES_DIR, TARGET_CSV_NAME)

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
EUROSTAT_OUT = os.path.join(DATA_DIR, "eurostat_gva_2022.csv")

# ── Zenodo record 3626071, Libraries.io v1.6.0 (Katz, 2020) ────────────
ZENODO_URL = (
    "https://zenodo.org/records/3626071/files/"
    "libraries-1.6.0-2020-01-12.tar.gz"
)
ZENODO_SIZE = 24_890_021_718
ZENODO_MD5 = "4f2275284b86827751bb31ce74238b15"

# ── Eurostat ───────────────────────────────────────────────────────────
EUROSTAT_YEAR = 2022
EUROSTAT_AGGREGATES = ["EU27_2020", "EU28", "EA19", "EA20", "EU15"]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download_tarball(force: bool) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    if force and os.path.exists(TARBALL_PATH):
        print("[zenodo] --force: removing existing tarball")
        os.remove(TARBALL_PATH)

    # Zenodo often takes 1–5 min to begin streaming a 25 GB file and may drop
    # long-running connections. Retry on timeouts/connection errors; the Range
    # header on subsequent attempts resumes from the current on-disk size.
    # timeout = (connect_timeout, read_timeout) per requests docs.
    CONNECT_TIMEOUT = 60
    READ_TIMEOUT = 300
    MAX_ATTEMPTS = 5

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if os.path.exists(TARBALL_PATH):
            size = os.path.getsize(TARBALL_PATH)
            if size == ZENODO_SIZE:
                print(f"[zenodo] Tarball already present ({human(size)}), skipping download")
                break
            start_byte = size
            if attempt == 1:
                print(f"[zenodo] Partial tarball on disk: "
                      f"{human(size)} / {human(ZENODO_SIZE)}; resuming")
        else:
            start_byte = 0

        headers = {"Range": f"bytes={start_byte}-"} if start_byte else {}
        print(f"[zenodo] GET {ZENODO_URL}"
              + (f" (attempt {attempt}/{MAX_ATTEMPTS})" if attempt > 1 else ""))

        try:
            with requests.get(
                ZENODO_URL, stream=True, headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as r:
                r.raise_for_status()
                mode = "ab" if start_byte else "wb"
                with open(TARBALL_PATH, mode) as f, tqdm(
                    total=ZENODO_SIZE, initial=start_byte, unit="B",
                    unit_scale=True, unit_divisor=1024,
                    desc="[zenodo] download", mininterval=1.0,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))
            break  # finished without exception
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt == MAX_ATTEMPTS:
                sys.exit(f"ERROR: download failed after {MAX_ATTEMPTS} attempts: {e}")
            wait = min(2 ** attempt, 60)
            print(f"[zenodo] transfer interrupted ({type(e).__name__}); "
                  f"retrying in {wait}s (resume from "
                  f"{human(os.path.getsize(TARBALL_PATH) if os.path.exists(TARBALL_PATH) else 0)})")
            import time
            time.sleep(wait)

    size = os.path.getsize(TARBALL_PATH)
    if size != ZENODO_SIZE:
        sys.exit(f"ERROR: downloaded size {size:,} != expected {ZENODO_SIZE:,}. "
                 f"Re-run (the script will resume) or pass --force.")

    md5 = hashlib.md5()
    with open(TARBALL_PATH, "rb") as f, tqdm(
        total=size, unit="B", unit_scale=True, unit_divisor=1024,
        desc="[zenodo] MD5 verify", mininterval=1.0,
    ) as bar:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            md5.update(chunk)
            bar.update(len(chunk))
    actual = md5.hexdigest()
    if actual != ZENODO_MD5:
        sys.exit(f"ERROR: MD5 mismatch (got {actual}, expected {ZENODO_MD5}). "
                 f"Delete {TARBALL_PATH} and rerun.")
    print(f"[zenodo] MD5 OK: {actual}")


def extract_target_csv(force: bool) -> None:
    if (not force) and os.path.exists(TARGET_CSV_PATH):
        size = os.path.getsize(TARGET_CSV_PATH)
        print(f"[extract] {TARGET_CSV_NAME} already present ({human(size)}), skipping")
        return

    if not os.path.exists(TARBALL_PATH):
        sys.exit(f"ERROR: tarball not found at {TARBALL_PATH}. "
                 f"Run without --skip-zenodo or remove --skip-zenodo to download it.")

    os.makedirs(LIBRARIES_DIR, exist_ok=True)
    tar_size = os.path.getsize(TARBALL_PATH)
    print(f"[extract] Streaming tarball to locate {TARGET_CSV_NAME}...")
    # Streaming mode cannot seek, but walking members in order lets us skip
    # the ~134 GB of files preprocess.py does not read. Progress tracks
    # compressed bytes read from disk (which is why the bar may finish early
    # when we find the target before the end of the archive).
    with open(TARBALL_PATH, "rb") as raw:
        wrapped = tqdm.wrapattr(
            raw, "read", total=tar_size, unit="B", unit_scale=True,
            unit_divisor=1024, desc="[extract] scan", mininterval=1.0,
        )
        with tarfile.open(fileobj=wrapped, mode="r|gz") as tar:
            for member in tar:
                if member.isfile() and os.path.basename(member.name) == TARGET_CSV_NAME:
                    wrapped.write(f"\n[extract] Found {member.name} "
                                  f"({human(member.size)}); writing\n")
                    src = tar.extractfile(member)
                    if src is None:
                        sys.exit(f"ERROR: could not open {member.name} inside tarball")
                    with open(TARGET_CSV_PATH, "wb") as dst, tqdm(
                        total=member.size, unit="B", unit_scale=True,
                        unit_divisor=1024, desc="[extract] write",
                        mininterval=1.0,
                    ) as wbar:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)
                            wbar.update(len(chunk))
                    print(f"[extract] Wrote {TARGET_CSV_PATH} "
                          f"({human(os.path.getsize(TARGET_CSV_PATH))})")
                    return
    sys.exit(f"ERROR: {TARGET_CSV_NAME} not found in tarball")


def fetch_eurostat(force: bool) -> None:
    if (not force) and os.path.exists(EUROSTAT_OUT):
        print(f"[eurostat] Cache exists at {EUROSTAT_OUT}, skipping fetch")
        return

    try:
        from dbnomics import fetch_series
    except ImportError:
        sys.exit("ERROR: dbnomics not installed. Run: pip install -r requirements.txt")

    print("[eurostat] Fetching nama_10_a64 (J62_J63, B1G, CP_MEUR) via DBnomics")
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
    print(f"[eurostat] Wrote {EUROSTAT_OUT} "
          f"({len(gva)} countries, total = EUR {gva['gva_meur'].sum():,.0f}M)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--force", action="store_true",
                    help="Re-download, re-extract and re-fetch even if outputs exist.")
    ap.add_argument("--skip-zenodo", action="store_true",
                    help="Skip the Libraries.io tarball download and extraction.")
    ap.add_argument("--skip-eurostat", action="store_true",
                    help="Skip the Eurostat fetch.")
    ap.add_argument("--keep-tarball", action="store_true",
                    help="Retain the 25 GB tarball after extraction "
                         "(default: delete to reclaim disk).")
    args = ap.parse_args()

    if not args.skip_zenodo:
        download_tarball(force=args.force)
        extract_target_csv(force=args.force)
        if not args.keep_tarball and os.path.exists(TARBALL_PATH):
            os.remove(TARBALL_PATH)
            print(f"[zenodo] Removed tarball (reclaimed {human(ZENODO_SIZE)}). "
                  f"Pass --keep-tarball next time to retain it.")

    if not args.skip_eurostat:
        fetch_eurostat(force=args.force)

    print("[done] Data fetch complete. Run: python preprocess.py")


if __name__ == "__main__":
    main()
