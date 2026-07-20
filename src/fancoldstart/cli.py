"""Command-line entry points.

  python -m fancoldstart.cli smoke        run the synthetic validation pipeline
  python -m fancoldstart.cli wikidata     fetch tournament attendance from Wikidata
  python -m fancoldstart.cli version      print the version

Real collection (reddit, bluesky) is driven from scripts once credentials are set;
see the README. The smoke command needs no network and no keys.
"""
import argparse
import json

from . import __version__


def _cmd_smoke(args):
    from .collect.synthetic import generate
    from .pipeline import run
    events = generate(n_fans=args.n_fans, seed=args.seed)
    out = run(events, t_cutoff=args.cutoff, tau=args.tau,
              provenance="synthetic", seed=args.seed)
    print(out["report_md"])


def _cmd_wikidata(args):
    from .collect import wikidata
    matches = wikidata.fetch_matches()
    agg = wikidata.total_and_peak_attendance(matches)
    print(json.dumps(agg, indent=2))


def _cmd_version(args):
    print(__version__)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fancoldstart")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="run the synthetic validation pipeline")
    s.add_argument("--n-fans", type=int, default=400)
    s.add_argument("--cutoff", type=float, default=40.0)
    s.add_argument("--tau", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(func=_cmd_smoke)

    w = sub.add_parser("wikidata", help="fetch tournament attendance from Wikidata")
    w.set_defaults(func=_cmd_wikidata)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=_cmd_version)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
