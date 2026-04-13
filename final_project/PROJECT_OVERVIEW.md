# Project Overview: Deviations from Proposal and Operational Decisions

This document records every operational decision that deviates from or extends what was stated in `proposal.md`. It is intended as a companion to the proposal, not a replacement. The proposal describes what we planned; this document describes what we actually did and why.

---

## 1. Data Cleaning: CSV Off-by-One Column Shift

**Proposal assumption:** Libraries.io CSV can be read directly with pandas.

**What happened:** The `projects_with_repository_fields` CSV has 59 header columns but 60 data fields per row (trailing comma). Standard `pd.read_csv()` misaligns every column after the first. All numeric fields (dependent counts, contributor counts, stars) read as wrong values.

**Fix:** Read the header row separately with Python's `csv` module, append a dummy `_extra` column name, then pass the corrected header to pandas via `names=header, skiprows=1`. This is in `analysis.py:load_data()`.

---

## 2. Missing Contributor Imputation

**Proposal assumption:** Not discussed.

**What happened:** 28% of load-bearing packages have no contributor count in Libraries.io (packages without linked repository metadata).

**Decision:** Impute as 1 (single maintainer). This is a conservative/fragile assumption — it treats missing data as evidence of under-resourcing rather than excluding these packages. The alternative (dropping them) would remove over a quarter of the sample and bias toward well-documented packages.

---

## 3. Fragility Specification: Additive, Not Multiplicative

**Proposal:** Describes "weighted additive combinations" — this is what we implemented. However, an earlier iteration of the project (in `bigger_project/`) used a multiplicative specification that failed due to zero-inflation (the 75th percentile of fragility was indistinguishable from zero). The additive specification was adopted before the proposal was written, but the rationale is worth recording: any single zero component (e.g., zero open issues) kills the entire product, making most packages score zero on fragility.

---

## 4. Topic Modelling: NMF over LDA

**Proposal:** "LDA or NMF."

**Decision:** NMF exclusively. LDA assumes a generative probabilistic model that works well for long documents with rich vocabulary. Package descriptions are typically one sentence of technical jargon. NMF's non-negativity constraint matches TF-IDF's non-negative structure and produces sparser, more interpretable topics on this corpus. This is a standard finding in short-text topic modelling literature.

---

## 5. Topic Count: k=25

**Proposal:** Did not specify k.

**Process:** We tested k=5, 10, 15, 18, 20, 22, 25 through a combination of reconstruction error sweeps and manual inspection of topic quality.

| k | Largest cluster | Problem |
|---|----------------|---------|
| 10 | 1,980 (21%) | Single "core/package/api" catch-all held 40% of risk universe |
| 15 | 1,802 (19%) | Same catch-all, slightly smaller |
| 20 | 898 (9.5%) | Catch-all breaking up, topics becoming interpretable |
| 25 | 927 (9.8%) | No cluster >10%, topics are distinct functional categories |

**Decision:** k=25. The reconstruction error curve shows no sharp elbow (monotonically decreasing from 92.1 at k=5 to 87.4 at k=25). Selection was based on topic quality: at k=25 the topics resolve into recognisable functional categories (HTTP/networking, JSON/serialisation, CSS/styling, cloud API clients, test runners, etc.) and the AWS/SDK cluster (Topic 5: "client, google, apis, rest") emerges as a distinct group with meaningful fragility characteristics.

---

## 6. Domain-Specific Stop Words and URL Stripping

**Proposal:** Did not discuss stop words.

**Problem:** The default sklearn English stop word list (318 words) is designed for natural language, not for software package descriptions. Without domain-specific filtering:
- Topics were dominated by artifact-type terms: "package", "library", "module", "plugin"
- Platform identifiers appeared as topic keywords: "npm", "node", "js", "java"
- Badge/shield image URLs from markdown descriptions polluted the vocabulary: "shields", "svg", "dm", "downloads"
- URL fragments appeared as top terms: "https", "com", "github", "io"

**Fix (two-part):**

1. **URL stripping** before tokenisation: regex removal of `https?://` URLs, markdown image syntax `![...](...)`, markdown links `[text](url)` (keeping text), and HTML tags `<...>`. This is applied in `_clean_descriptions()`.

2. **Domain stop word list** (80+ terms) covering:
   - Artifact types: package, library, module, plugin, wrapper, helper, utility, sdk, api
   - Platform identifiers: npm, node, js, javascript, php, python, ruby, java, net, asp
   - Marketing filler: simple, lightweight, fast, easy, modern, powerful, flexible, robust
   - Generic software terms: code, project, application, tool, implementation, core, base
   - Generic structure terms: file, data, type, method, function, class, object, config
   - Generic programming terms: string, stream, buffer, browser, convert, build, run, create, async, event
   - URL/badge residuals: cross, platform, like, runtime, directory

**Impact:** TF-IDF vocabulary went from 1,933 features to 1,755 features. The largest cluster dropped from 21% (k=10, no domain stops) to 9.8% (k=25, with domain stops + URL stripping). Topic labels became functional rather than artifactual.

---

## 7. Criticality: Libraries.io Only, No OpenSSF

**Proposal:** Did not mention OpenSSF.

**Context:** An earlier iteration (in `bigger_project/`) incorporated the OpenSSF Criticality Score at 10% weight (60/30/10 split). For the submission we dropped it to keep the pipeline self-contained — the OpenSSF data requires a separate 154MB download from Google Cloud Storage and is not part of the Libraries.io dataset.

**Consequence:** Criticality is driven entirely by Libraries.io dependent counts (65% package dependents, 35% repository dependents). npm coverage is more complete than Maven or NuGet because enterprise Java/.NET deployments often use private artifact repositories that don't report to public registries. The npm-heavy composition of the risk universe (154/247 = 62%) partly reflects this coverage asymmetry. Acknowledged as a limitation.

---

## 8. Risk Universe Threshold: 75th Percentile

**Proposal:** States "packages above the 75th percentile on both dimensions."

**Justification (not in proposal):** The 75th percentile is a conventional choice analogous to percentile thresholds in financial Value-at-Risk analysis. At this threshold, if C and F were independent, 6.25% of packages would fall in the risk universe. The actual figure is 2.6% (247/9,461), reflecting the negative correlation between C and F (r = -0.31). The threshold sensitivity analysis (Stage 4d) shows the risk universe ranges from 12 (90th) to 1,146 (60th) packages. As a sanity check, `log4j:log4j` (the package responsible for the December 2021 log4shell vulnerability) falls within the risk universe at the 75th percentile threshold, with C=0.625 and F=0.903.

---

## 9. Criticality-Fragility Correlation: A Finding, Not an Assumption

**Proposal:** Did not discuss.

**Finding:** C and F are negatively correlated (Pearson r = -0.31, Spearman rho = -0.26, both p < 1e-140). More critical packages tend to be less fragile. This means the market partially works — visibility and downstream importance attract maintenance investment. The 247 packages in the risk universe are the exceptions: they have somehow avoided the attention that their dependency count would normally attract. This reframes the economics: the question is not "why is OSS fragile?" (it mostly isn't, for critical packages) but "what characterises the packages that slip through?"

---

## 10. LASSO Validation: Partial Circularity Acknowledged

**Proposal:** States LASSO will validate fragility weights.

**Complication:** The LASSO target is bus_factor_1 (contributor count < 2), which is itself one of the four fragility components (weight: 20%). This creates mild circularity. We use the LASSO as a directional validation (do the indicators co-move with an observable proxy of under-resourcing?), not to set the weights.

**Result:** Stars/popularity is the strongest predictor (coefficient -0.29): popular packages attract more contributors. Staleness was zeroed out by LASSO — it doesn't independently predict bus-factor-1 after controlling for other features. R² = 0.24 in 5-fold CV — modest but meaningful for a binary outcome predicted from continuous features.

---

## 11. Topic 0 Composition: A Residual Cluster, Not a TypeScript Cluster

**Auto-generated NMF keywords:** `typescript, redux, chai, cookie` — these are the four highest-weight TF-IDF terms in the topic-term matrix, and on first reading they suggest a coherent JavaScript-ecosystem cluster. They are misleading.

**Human-readable label assigned:** `"Generic/legacy infrastructure (residual)"`. This is the label used in the report, the summary tables, all figures, and `outputs/cluster_summary.csv`. The auto-generated keyword list is preserved in the notebook output for transparency but is never used as a finding label.

**Why the relabel:** NMF assigned 927 packages to Topic 0 — the second-largest cluster — but the membership is heterogeneous in a specific way. The cluster is dominated by *packages with uninformative descriptions* across every ecosystem in the dataset:

| Platform | Topic 0 packages |
|---|---|
| NPM | 740 |
| Maven | 111 |
| Packagist | 36 |
| Pypi | 15 |
| Rubygems | 14 |
| NuGet | 9 |
| Cargo | 2 |

NPM is the modal platform by count, but the **high-fragility tail is overwhelmingly Maven**. Of the top-15 most fragile packages in Topic 0, 14 are legacy Java/Maven infrastructure: `aopalliance` (F=0.93), `javax.persistence:persistence-api` (F=0.92), `jdom` (F=0.92), `plexus-utils` (F=0.93), `concurrent:concurrent` (F=0.93), `xstream` (F=0.92), `easymock` (F=0.92) — all single-maintainer, deeply embedded enterprise Java dependencies. These are genuinely fragile runtime infrastructure that NMF could not assign to a more specific topic because their package descriptions are short and generic.

**On the `@types/*` concern:** An initial worry was that the 314 `@types/*` DefinitelyTyped packages in Topic 0 were inflating fragility. The opposite is true: their mean fragility (F=0.225) is below the cluster mean because they inherit the DefinitelyTyped monorepo's ~1,100 contributors. They pull Topic 0's mean *down*. No filtering is needed.

**How the finding is framed in the report:** The most fragile cluster is "generic/legacy infrastructure" — a residual category whose risk-bearing members are predominantly legacy Java/Maven dependencies with uninformative descriptions, not the TypeScript ecosystem the auto-keywords would suggest. The lesson for methodology is that NMF top-keyword labels can mis-frame a residual cluster, and any finding built on them must be cross-checked against the actual high-weight package list. Topic 24 ("General-purpose infrastructure (residual)") is treated symmetrically.

---

## 12. Threshold Sensitivity: Ranking Is Unstable

**Proposal:** States sensitivity analysis will be performed.

**Finding:** The most fragile cluster *by mean fragility* at the 75th percentile is Topic 0, "Generic/legacy infrastructure (residual)" (F=0.38) — see §11 for why this label, not the auto-generated `typescript/redux/chai` keywords. But the cluster with the highest *risk concentration* (share of cluster members in the risk universe) is a different one and varies across thresholds: "Command-line infrastructure" at the 60th, "Task runners (Gulp/Rake)" at the 75th, "Cloud & API clients" at the 90th. This is an honest finding — the functional category that "looks worst" depends on which question you ask (mean fragility vs. risk concentration) and which percentile threshold you set. The report discusses this as a limitation rather than hiding it.

---

## 13. GVA as Context, Not Calibrated VaR

**Proposal:** States GVA provides "the economic denominator."

**Decision:** Eurostat GVA (EUR 467B for J62/J63 in 2022) is used as economic context only — "how large is the software sector above this infrastructure?" The funding gap (EUR 22.2M) is presented in absolute terms per cluster. An earlier iteration calibrated against log4shell remediation cost to produce a euro-denominated VaR, but this required a circular calibration anchor and was dropped as overly complex for a course project.

---

## 14. Funding Gap: A Lower Bound, Not a Point Estimate

**Proposal:** States "maintenance funding gap is estimated per functional cluster."

**Headline framing:** **The EUR 22.2M figure is a lower bound, not a calibrated point estimate.** It should be read as "the funding gap is at least this large," and the cluster-level breakdown matters more than the headline total. The bootstrap CI [19.7, 24.8] characterises *sampling uncertainty in the risk universe size only* — it does not quantify any of the specification uncertainty below.

**Specification:** 2 FTEs per risk-universe package at the EU median software developer salary (EUR 45,000/year = EUR 90,000 per package per year). The 2-FTE floor is drawn from OpenSSF guidance on minimum viable maintenance staffing (security-response capacity + bus-factor redundancy).

**Why this is a lower bound — four reasons that all push in the same direction:**

1. **Salary is below market.** EUR 45,000 is the EU-wide median; critical infrastructure maintainers in Western Europe command EUR 80,000–120,000. Using market rate would raise the gap by roughly 1.8×–2.7×.
2. **Uniform staffing under-funds critical packages.** A 2-FTE floor is appropriate for a small utility but inadequate for log4j-class infrastructure. Variable staffing weighted by criticality would raise the gap further.
3. **No existing funding is subtracted.** The figure is gross, not net of corporate sponsorship, foundation grants, or de-facto employer-paid maintenance time. Net unmet need is unknown but strictly smaller; gross need is what the figure captures.
4. **Risk universe is itself a floor.** The 75th-percentile threshold means 247 packages are flagged; loosening to the 60th percentile flags 1,146 (see Stage 4d). The chosen threshold is conservative.

**What the figure is good for:**
- Order-of-magnitude reasoning ("tens of millions of euros, not hundreds or billions").
- Per-cluster comparison (Topic 0 holds €1.4M of the gap, Topic 14 holds €1.4M, etc. — see `outputs/funding_gap_by_cluster.csv`).
- Comparison against EU policy spend on OSS — Sovereign Tech Fund's 2023 budget was EUR 11.5M, NLnet's annual disbursement is in the low millions; €22M is the same order of magnitude as existing public funding combined, suggesting the unmet need is roughly equal to the entire public-funding ecosystem.

**What the figure is *not* good for:** any precise euro claim, any benefit-cost ratio against avoided incident cost, any single-package funding recommendation.

---

## Summary of Key Numbers

| Metric | Value |
|--------|-------|
| Packages in dataset | 4.6 million |
| Target ecosystems | 7 (npm, Maven, PyPI, NuGet, RubyGems, Packagist, Cargo) |
| Load-bearing packages (>=100 dependents) | 9,461 |
| Risk universe (75th pctile on C and F) | 247 |
| C-F correlation (Pearson r) | -0.31 |
| NMF topics (k) | 25 |
| Largest cluster | 927 (9.8%) |
| Most fragile cluster (mean F) | Topic 0: Generic/legacy infrastructure — residual; high-fragility tail is legacy Java/Maven (F=0.38, 27% bus-factor=1) |
| Highest risk concentration (75th pctile) | Topic 14: Task runners (Gulp/Rake) |
| EU J62/J63 GVA (2022) | EUR 467B |
| Total funding gap (gross lower bound) | ≥ EUR 22.2M [95% CI: 19.7, 24.8 — sampling uncertainty only; see §14] |
| Bootstrap replicates | 1,000 |
| LASSO CV R² | 0.24 |
