# Raw Data Sources

This directory holds the unmodified upstream inputs for the project. None of
these files are produced by code here — they must be obtained from their
original sources. The top-level `fetch_data.py` script automates the download
and staging; this README documents provenance and provides a manual fallback.

## 1. Libraries.io v1.6.0 (Zenodo)

Path expected: `raw_data/libraries-1.6.0-2020-01-12/projects_with_repository_fields-1.6.0-2020-01-12.csv`

**Source:**

> Katz, J. (2020). *Libraries.io Open Source Repository and Dependency Metadata* (1.6.0). Zenodo.
> https://doi.org/10.5281/zenodo.3626071

The Zenodo record ships as a single 24.89 GB `tar.gz` bundle (~159 GB
uncompressed) containing seven CSVs. Only one is consumed by `preprocess.py`:

- `projects_with_repository_fields-1.6.0-2020-01-12.csv` (~25 GB)

The other members (`dependencies-`, `repositories-`, `versions-`, `tags-`, etc.)
are not read by any script here and can be ignored.

### Automated fetch (recommended)

```bash
python fetch_data.py
```

Behaviour:
- downloads `libraries-1.6.0-2020-01-12.tar.gz` from Zenodo (resume-aware,
  MD5-verified against `4f2275284b86827751bb31ce74238b15`),
- stream-extracts **only** `projects_with_repository_fields-*.csv` into
  `raw_data/libraries-1.6.0-2020-01-12/`,
- deletes the tarball after extraction (pass `--keep-tarball` to retain it).

Peak disk usage: ~50 GB (tarball + extracted CSV briefly coexist); steady
state: ~25 GB.

### Manual fallback

If you'd rather fetch by hand (e.g. behind a proxy):

```bash
mkdir -p raw_data/libraries-1.6.0-2020-01-12
curl -L -o raw_data/libraries-1.6.0-2020-01-12.tar.gz \
    https://zenodo.org/api/records/3626071/files/libraries-1.6.0-2020-01-12.tar.gz/content

# Extract only the one file the pipeline needs.
tar -xzvf raw_data/libraries-1.6.0-2020-01-12.tar.gz \
    -C raw_data/libraries-1.6.0-2020-01-12 --strip-components=1 \
    libraries-1.6.0-2020-01-12/projects_with_repository_fields-1.6.0-2020-01-12.csv

rm raw_data/libraries-1.6.0-2020-01-12.tar.gz
```

Alternatively, a symlink to an existing local copy is equivalent:

```bash
ln -s /path/to/your/libraries-1.6.0-2020-01-12 raw_data/libraries-1.6.0-2020-01-12
```

## 2. EU Commission Open Source Study (2021)

File: `CNECT_OpenSourceStudy_EN_28_6_2021_*.pdf`

Background reference cited in the report. Not consumed by code. Source:

> Blind, K., et al. (2021). *The impact of Open Source Software and Hardware on technological independence, competitiveness and innovation in the EU economy.* European Commission, DG CNECT.
> https://op.europa.eu/en/publication-detail/-/publication/29effe73-2c2c-11ec-bd8e-01aa75ed71a1

## 3. Eurostat GVA (not stored in raw_data/)

`fetch_data.py` retrieves the National Accounts series `nama_10_a64`
(NACE J62_J63 Computer programming, consultancy, data processing — Gross Value
Added at current prices, EUR million) via the DBnomics API and writes a cached
copy to `data/eurostat_gva_2022.csv`. To refresh the cache, pass `--force` to
`fetch_data.py`.
