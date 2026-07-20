"""Run the pre-registered pipeline on the collected REAL World Cup data and save
the findings, with full provenance, for the dashboard.

Reads data/wc2026_reddit_events.jsonl (produced by scripts/collect_real.py) and
data/provenance.json, runs the pipeline with a temporal calibration/holdout split
inside the tournament window, and writes data/real_results.json. Every number is
computed from the real archived comments. Nothing is fabricated. This is a pilot
on a bounded real sample, not the full population, and it uses the Reddit graph
only; the Bluesky live window was not captured.

    python scripts/run_real.py [cutoff_day] [tau_days]
"""
import sys
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fancoldstart.eval.splits import make_splits  # noqa: E402
from fancoldstart.pipeline import run  # noqa: E402


def main(cutoff=24.0, tau=15.0):
    data = ROOT / "data"
    events = [json.loads(l) for l in open(data / "wc2026_reddit_events.jsonl", encoding="utf-8")]
    prov = json.loads((data / "provenance.json").read_text(encoding="utf-8"))

    splits = make_splits(events, cutoff, tau, 1)
    freq = np.bincount(splits.x.astype(int), minlength=7)
    hist = [int(v) for v in freq[:6]] + [int(freq[6:].sum())]

    out = run(events, t_cutoff=cutoff, tau=tau, provenance="real", seed=7)

    # community sizes from the calibration window
    comm = {}
    for e in events:
        if e["t"] <= cutoff:
            comm[e["community"]] = comm.get(e["community"], 0) + 1

    findings = {
        "provenance": prov,
        "split": {"cutoff_day": cutoff, "tau_days": tau,
                  "window": "11 Jun to 19 Jul 2026"},
        "n_fans": out["n_fans"], "n_cold": out["n_cold"],
        "pct_cold": round(100.0 * out["n_cold"] / max(out["n_fans"], 1), 1),
        "freq_hist": hist,
        "communities": comm,
        "bgnbd": {k: round(v, 4) for k, v in out["params"].items()},
        "results": [
            {"name": r["name"], "primary": r["primary"], "effect": r["effect"],
             "effect_name": r["effect_name"], "p": r["p"], "p_bh": r["p_bh"],
             "supported": r["supported"], "note": r["note"]}
            for r in out["results"]
        ],
    }
    (data / "real_results.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print(f"REAL run | fans={findings['n_fans']} cold={findings['n_cold']} "
          f"({findings['pct_cold']}%) | communities={len(comm)}")
    print(f"BG/NBD: {findings['bgnbd']}")
    for r in findings["results"]:
        print(f"  {r['name']} eff={r['effect']:+.4f} p={r['p']:.4f} supported={r['supported']}")
    print("saved data/real_results.json")


if __name__ == "__main__":
    a = sys.argv
    main(float(a[1]) if len(a) > 1 else 24.0, float(a[2]) if len(a) > 2 else 15.0)
