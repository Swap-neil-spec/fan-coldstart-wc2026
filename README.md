# fan-coldstart-wc2026

Predicting fan future engagement under sparse signals: a pre-registered
cold-start design for customer-base analysis, run on free public data from the
2026 FIFA World Cup.

This is the reproducible pipeline for the working paper "Predicting Fan Future
Engagement Under Sparse Signals" by Swapnil (Neil) Rajkumar Gaikwad.

## Status

This is a design and pre-registration, not a results paper. No fitted numbers are
claimed here or in the paper. Every model output is produced by running this
pipeline on collected data. The pre-registration is in
[PREREGISTRATION.md](PREREGISTRATION.md).

## The idea

Customer-base analysis (BG/NBD, Pareto/NBD) predicts a person's future activity
from their own sparse history, and it breaks at the cold start, where a newly
arrived fan has almost no history to condition on. A World Cup creates that
problem at scale: a dated influx of low-history fans over a few weeks. This
pipeline extends the BG/NBD backbone into the cold-start regime by borrowing
predictive signal from a fan's position in the public social graph, through a
graph-smoothness prior and an inductive message-passing model. The design is
observational and makes predictive, not causal, claims: co-participation edges
encode homophily, so graph position is used only as a predictive prior.

## What the pipeline does

Events to graph to features to splits to models to pre-registered tests to report:

- builds the fan graph from Reddit co-participation and reply edges, with Bluesky
  as a complementary live source and Wikidata for match and attendance facts
- computes public-data position features: community breadth, weak-tie bridging,
  neighborhood density, and activity
- fits a BG/NBD baseline, a graph-smoothed latent-attrition model, and an
  inductive cold-start head
- runs the pre-registered hypotheses H1 to H7 with the fixed tests, alpha 0.05,
  Benjamini-Hochberg correction, and pre-registered smallest effect sizes
- emits a report labeled with its data provenance

## Install

Core (runs the pipeline, the validation, and the tests; numpy and networkx only):

    pip install -r requirements-core.txt

Full (adds live collectors and config):

    pip install -r requirements.txt

The special functions, optimizer, and generalized linear models are implemented
on numpy, so the science reproduces with a small, wheel-only dependency set and
does not require scipy or statsmodels.

## Quickstart

Run the synthetic validation pipeline (no network, no keys):

    python scripts/run_smoke.py

Run the tests:

    pytest

The synthetic run uses labeled synthetic data to confirm the pipeline executes
end to end and that the tests have power. Its numbers are not research results and
appear nowhere in the paper. The report it prints is banner-labeled as synthetic.

## Real data collection

1. Set Reddit credentials in the environment: `REDDIT_CLIENT_ID`,
   `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (approved non-commercial tier).
2. Edit [config/sources.yaml](config/sources.yaml) for subreddits, keywords, the
   calibration window, and the horizon.
3. Collect: Reddit (`fancoldstart.collect.reddit`), Bluesky Jetstream
   (`fancoldstart.collect.bluesky`), Wikidata (`fancoldstart.collect.wikidata`).
4. Run the pipeline on the collected events with `fancoldstart.pipeline.run`,
   passing `provenance="real"`.

Raw pulls should be stored under `data/` (git-ignored) with retrieval timestamps.
Reddit and Bluesky data are not redistributable, so this repository ships the
collectors, not the corpus. See [DATA_LICENSES.md](DATA_LICENSES.md).

## Hypotheses

H1 cold-start lift from the graph prior; H2 effect concentration; H3 community
breadth predicts persistence; H4 weak-tie bridging predicts engagement; H5
prediction beyond activity (the load-bearing falsification, via degree-preserving
rewiring); H6 inductive cold-start transfer; H7 cross-platform robustness. Full
statements, tests, and thresholds are in [PREREGISTRATION.md](PREREGISTRATION.md).

## Repository structure

    src/fancoldstart/
      special.py, optimize.py, stats.py, glm.py   numerics on numpy + stdlib
      collect/      reddit, bluesky, wikidata collectors + synthetic generator
      graph/        fan-graph build and position features
      models/       bgnbd, graph_smoothed, inductive
      eval/         splits, hypotheses (H1..H7), report
      pipeline.py   end-to-end orchestration
      cli.py        command-line entry points
    tests/          special, bgnbd, glm, features, smoke
    config/         sources.yaml
    scripts/        run_smoke.py

## Notes on the models

The BG/NBD likelihood, conditional expectation, and predictions follow Fader,
Hardie and Lee (2005). The graph-smoothed model is random-walk-normalized
Laplacian regularization of the per-fan log-rate, solved by sparse Jacobi
iteration, so it scales without forming a dense Laplacian. The inductive head is a
linear GraphSAGE-mean with a ridge solution; a deeper PyTorch Geometric version is
a drop-in replacement and is not required for the claim. The graph-smoothing
strength gamma is tuned by temporal cross-validation on real data.

## Citing

See [CITATION.cff](CITATION.cff).

## Author

Swapnil (Neil) Rajkumar Gaikwad.
