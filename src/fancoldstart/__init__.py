"""
fancoldstart: a reproducible pipeline for predicting fan future engagement
under sparse signals, using social-graph structure as a predictive prior.

This package accompanies the pre-registered design paper
"Predicting Fan Future Engagement Under Sparse Signals" (Gaikwad, 2026).

Design principles enforced in code:
- No fabricated data. Collectors read only real, free, public sources.
- The only bundled data is a clearly labeled SYNTHETIC generator used to
  validate that the pipeline executes end to end. Synthetic output is never
  a research result and is labeled as such wherever it is produced.
- Model outputs on real data are produced by running this pipeline, not
  asserted in the paper.
"""

__version__ = "0.1.0"
