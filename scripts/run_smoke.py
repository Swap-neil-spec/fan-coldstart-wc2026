"""Run the synthetic validation pipeline and print the report.

    python scripts/run_smoke.py

This needs no network and no API keys. It confirms the pipeline executes end to
end and that the pre-registered tests run. The numbers are from labeled synthetic
data and are not research results.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fancoldstart.collect.synthetic import generate  # noqa: E402
from fancoldstart.pipeline import run  # noqa: E402


def main():
    events = generate(n_fans=400, seed=13)
    out = run(events, t_cutoff=40.0, tau=20.0, provenance="synthetic", seed=13)
    print(out["report_md"])


if __name__ == "__main__":
    main()
