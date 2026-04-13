# Raw Data Sources

This directory contains the unmodified upstream inputs for the project. None of these files are produced by code in this repository — they must be obtained from their original sources before `preprocess.py` can run.

## 1. Libraries.io v1.6.0 (Zenodo)

Path expected: `raw_data/libraries-1.6.0-2020-01-12/`

The full snapshot is **~159 GB uncompressed**, so it is not bundled. Download from:

> Katz, J. (2020). *Libraries.io Open Source Repository and Dependency Metadata* (1.6.0). Zenodo.
> https://doi.org/10.5281/zenodo.3626071

Only one file from this snapshot is consumed by `preprocess.py`:

- `projects_with_repository_fields-1.6.0-2020-01-12.csv` (~25 GB)

Other CSVs in the tarball (`dependencies-`, `repositories-`, etc.) are unused by the current pipeline and may be safely omitted.

In the local environment, this directory is a **symlink** pointing to a copy stored outside the project root to keep the repository light. To recreate the symlink:

```bash
ln -s /path/to/your/libraries-1.6.0-2020-01-12 raw_data/libraries-1.6.0-2020-01-12
```

## 2. EU Commission Open Source Study (2021)

File: `CNECT_OpenSourceStudy_EN_28_6_2021_*.pdf`

Background reference cited in the report. Not consumed by code. Source:

> Blind, K., et al. (2021). *The impact of Open Source Software and Hardware on technological independence, competitiveness and innovation in the EU economy.* European Commission, DG CNECT.
> https://op.europa.eu/en/publication-detail/-/publication/29effe73-2c2c-11ec-bd8e-01aa75ed71a1

## 3. Eurostat GVA (fetched at runtime, then cached)

Not stored under `raw_data/`. The first run of `preprocess.py` fetches the National Accounts series `nama_10_a64` (NACE J62_J63 Computer programming, consultancy, data processing — Gross Value Added at current prices, EUR million) via the DBnomics API and writes a cached copy to `data/eurostat_gva_2022.csv`. Subsequent runs use the cache.

To force a refresh, delete the cached file before running `preprocess.py`.
