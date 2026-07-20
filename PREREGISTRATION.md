# Pre-registration

Author: Swapnil (Neil) Rajkumar Gaikwad. This document fixes the design, the
hypotheses, and the analysis before any model is fit on real data. It is the
registration referenced by the paper "Predicting Fan Future Engagement Under
Sparse Signals." No results exist at registration time.

## Study description
An observational design testing whether a fan's position in the public social
graph improves prediction of future engagement in the cold-start regime, where a
fan's own history is too thin for standard customer-base analysis. The backbone
is a BG/NBD-style latent-attrition model; the extension is a hierarchical model
with a graph-smoothness prior over the fan graph, plus an inductive message-
passing model for fans held out entirely from training. The claim is predictive,
not causal: co-participation edges encode homophily, so graph position is used
only as a predictive prior, and no structural or causal effect is asserted.

## Data
Collected live during the 2026 FIFA World Cup, 11 June to 19 July 2026, from free
public sources: the official Reddit Data API (formal fan graph, approved
non-commercial tier), the Bluesky AT Protocol Jetstream firehose (complementary
posts and follow graph), and Wikidata and Wikipedia (match structure, venues,
attendance). Historical baselines from Arctic Shift and a 2022 Twitter corpus.
Retrieval timestamps and collection gaps are logged. No sample size is fixed in
advance; the population is the observed fan influx.

## Cold-start definition
The cold-start stratum is fans with at most one prior engagement event in the
calibration window.

## Hypotheses
Primary family: H1, H3, H4, H5, H6. Secondary: H2, H7.

- H1. In the cold-start stratum, the graph-smoothed model achieves lower holdout
  forecast error for future engagement than the plain BG/NBD baseline.
- H2. The predictive benefit of the graph term declines monotonically as
  calibration-window frequency increases and is indistinguishable from zero for
  data-rich fans.
- H3. Higher community breadth (entropy of distinct communities engaged) predicts
  higher probability of remaining active into the holdout window, controlling for
  activity. Predictive, not structural.
- H4. Fans whose ties bridge otherwise separate communities show higher expected
  future engagement than equally active fans in a single dense community.
- H5. The predictive gains of H1, H3, and H4 survive activity controls, feature
  residualization, and degree-preserving edge rewiring. This is the load-bearing
  falsification test for the predictive claim.
- H6. The inductive message-passing model produces useful estimates for fans held
  out entirely from training, beating the population-average fallback that plain
  BG/NBD supplies for a fan with no usable individual signal.
- H7. The direction of H3 and H4 replicates in the 2022 Twitter supplement.

## Analysis
Fixed before fitting:

- Tests. H1 and H6: paired Wilcoxon signed-rank on holdout mean absolute error,
  with Spearman rank correlation reported alongside. H3: Wald test on the
  residualized breadth coefficient in a logistic persistence regression. H4: Wald
  test on the residualized bridging coefficient in a negative-binomial regression
  for holdout engagement counts, with activity and embeddedness controls. H2:
  interaction coefficient in a pooled error regression. H5: persistence of gains
  under the three robustness operations. H7: same-sign, same-significance
  replication in the 2022 corpus.
- Significance. Two-sided, alpha 0.05.
- Multiplicity. Benjamini-Hochberg across the primary family at false-discovery
  rate 0.05.
- Smallest effect sizes of interest. A five percent reduction in holdout mean
  absolute error for H1 and H6; a standardized coefficient of at least 0.1 in
  absolute value for the position predictors in H3 and H4. A result below its
  smallest effect size counts as null even if significant.

## What is not claimed
No fitted numbers appear in the paper or this registration. Every model output,
including the parameter-recovery simulation, is produced by running the released
pipeline on the collected data. The registration is committed before any fitting.
