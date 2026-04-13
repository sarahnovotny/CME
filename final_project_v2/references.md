# References

Working bibliography for the final report. Each entry includes a one-line note on what it supports in the argument so citations can be placed without re-reading the sources.

---

## Theoretical framing — common-pool resources, not pure public goods

**Ostrom, E.** (1990). *Governing the Commons: The Evolution of Institutions for Collective Action.* Cambridge University Press. https://doi.org/10.1017/CBO9780511807763
> The primary theoretical anchor for the paper. OSS infrastructure is better modelled as a common-pool resource (rivalrous in maintainer attention, partly excludable in practice, historically governed by community institutions) than as a pure Samuelsonian public good. Ostrom's design principles for stable CPR governance (clear boundaries, monitoring, graduated sanctions, nested enterprises) explain why the market *partly* works — the C–F correlation of −0.31 is consistent with self-organising community governance succeeding for visible packages and failing for the residual 247. Cite in the introduction when motivating the framework, and again in the discussion when arguing the funding-gap policy implication is "targeted backstop where governance fails," not "state provision."

**Schweik, C. M., & English, R. C.** (2012). *Internet Success: A Study of Open-Source Software Commons.* MIT Press. https://doi.org/10.7551/mitpress/9780262017251.001.0001
> Bridge from Ostrom's CPR framework to OSS specifically. Empirically classifies 174,000 SourceForge projects by governance outcome and finds that Ostrom's design principles predict project survival. Cite when extending the CPR framing to OSS infrastructure and when justifying the per-cluster funding analysis as a way to identify where the design principles have not held.

**Eghbal, N.** (2016). *Roads and Bridges: The Unseen Labor Behind Our Digital Infrastructure.* Ford Foundation.
https://www.fordfoundation.org/work/learning/research-reports/roads-and-bridges-the-unseen-labor-behind-our-digital-infrastructure/
> The qualitative complement to Schweik & English: documents the specific failure modes (single-maintainer burnout, deferred security work, no funded path from "useful side project" to "load-bearing infrastructure") that Ostrom's framework predicts when CPR governance is absent. Cite for the "infrastructure" frame and for the qualitative case that visibility ≠ funding.

**Nagle, F., Dana, J., Hoffman, J., Randazzo, S., & Zhou, Y.** (2022). *Census II of Free and Open Source Software — Application Libraries.* Linux Foundation & Harvard Laboratory for Innovation Science.
https://www.linuxfoundation.org/research/census-ii-of-free-and-open-source-software-application-libraries
> The empirical anchor for the "load-bearing OSS" concept and for the bus-factor / single-maintainer concern. Their methodology of identifying critical packages by downstream usage is the closest precedent for the criticality score used here.

---

## EU-specific evidence

**Blind, K., Böhm, M., Grzegorzewska, P., Katz, A., Muto, S., Pätsch, S., & Schubert, T.** (2021). *The impact of Open Source Software and Hardware on technological independence, competitiveness and innovation in the EU economy.* European Commission, DG CNECT. https://op.europa.eu/en/publication-detail/-/publication/29effe73-2c2c-11ec-bd8e-01aa75ed71a1
> Cited as the EU-policy backdrop for the funding-gap discussion. Establishes the order of magnitude of EU economic dependence on OSS and the policy interest in maintenance funding. PDF in `raw_data/`.

---

## Methodology — text as data

**Lee, D. D., & Seung, H. S.** (1999). "Learning the parts of objects by non-negative matrix factorization." *Nature*, 401(6755), 788–791. https://doi.org/10.1038/44565
> Origin paper for NMF. Cite once in the methods section when introducing the topic-modelling step.

**Sparck Jones, K.** (1972). "A statistical interpretation of term specificity and its application in retrieval." *Journal of Documentation*, 28(1), 11–21.
> Original IDF formulation. Cite alongside the NMF reference in the TF-IDF preprocessing description.

---

## Methodology — bootstrap inference

**Efron, B., & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC.
> Cite for the percentile-based 95% CIs in Stage 7 (Monte Carlo bootstrap, 1,000 replicates).

---

## Data sources (formal citations)

**Katz, J.** (2020). *Libraries.io Open Source Repository and Dependency Metadata* (1.6.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3626071
> The full snapshot used as the raw input to `preprocess.py`.

**Eurostat** (2024). *National accounts aggregates by industry (up to NACE A*64)* [`nama_10_a64`]. Retrieved via DBnomics. https://db.nomics.world/Eurostat/nama_10_a64
> Source for J62/J63 GVA at current prices, 2022. Cached in `data/eurostat_gva_2022.csv` after first fetch.

---

## Industry incident reference

**CISA & MITRE** (2021). "Apache Log4j Vulnerability Guidance" (CVE-2021-44228, "Log4Shell"). Cybersecurity & Infrastructure Security Agency. https://www.cisa.gov/news-events/news/apache-log4j-vulnerability-guidance
> Concrete example used as a sanity check (`log4j:log4j` falls inside the 75th-percentile risk universe with C=0.625, F=0.903) and as the canonical case of a critical-but-fragile package failing in production. Cite in the discussion when motivating why the cluster-level view matters operationally.

---

## OpenSSF Criticality Score (methodology reference)

**Open Source Security Foundation** (2022). *Criticality Score: A project that defines a criticality score for open source projects.* GitHub repository.
https://github.com/ossf/criticality_score
> Referenced in `PROJECT_OVERVIEW.md` §7 — earlier iteration used the OpenSSF score at 10% weight; dropped from the submitted pipeline to keep it self-contained. Cite when discussing the criticality-scoring choice and the limitation that this analysis uses Libraries.io dependent counts only.

---

## Notes on coverage gaps

- No formal econometric/causal-identification reference is included because the analysis is descriptive — there is no treatment-effect claim to defend. If the discussion section ends up making any conditional/causal statement, add a hedging citation or remove the claim.
- The funding-gap calibration (2 FTE × EUR 45 K) is acknowledged in `PROJECT_OVERVIEW.md` §14 as a lower bound; if a tighter cite for "minimum viable maintenance staffing" is needed, see OpenSSF's *Concise Guide for Developing More Secure Software* (2023) and the Sovereign Tech Fund grant guidelines (https://www.sovereigntechfund.de/programs/contribute).
