"""Collect a bounded, fully-sourced REAL sample of 2026 World Cup Reddit
discussion from the Arctic Shift archive (no auth), across several communities,
and run the pre-registered pipeline on it.

This is a real-data pilot, not the full population. Provenance is printed and
saved: the exact subreddits, the date window, per-subreddit caps, counts, and the
retrieval date. Bluesky live posts were not captured (the firehose window has
passed), so this pilot uses the Reddit co-participation and reply graph only.
Nothing is fabricated: every number comes from the archived comments.

    python scripts/collect_real.py
"""
import sys
import json
import time
import pathlib
import datetime as dt
from collections import defaultdict

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARCTIC = "https://arctic-shift.photon-reddit.com/api/comments/search"
UA = {"User-Agent": "fan-coldstart-wc2026/0.1 (research; Arctic Shift historical)"}

# Multiple communities so co-participation, breadth, and bridging are meaningful.
SUBREDDITS = ["soccer", "worldcup", "football", "USMNT", "MLS", "ussoccer",
              "reddevils", "Gunners", "chelseafc", "LiverpoolFC", "coys", "MCFC"]

# Tournament window: 11 June 2026 to 19 July 2026 (UTC).
ORIGIN = int(dt.datetime(2026, 6, 11, tzinfo=dt.timezone.utc).timestamp())
END = int(dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc).timestamp())
BUCKET_DAYS = 3                 # sample within each 3-day slice across the window
CAP_PER_BUCKET = 220            # comments per subreddit per time bucket
PAGE = 100


def pull_subreddit(sub):
    """Sample comments across the whole window by time bucket, so engaged fans
    who return across matches are captured rather than only opening-day authors."""
    rows = []
    bucket = BUCKET_DAYS * 86400
    start = ORIGIN
    while start < END:
        stop = min(start + bucket, END)
        after = start
        got = 0
        while got < CAP_PER_BUCKET and after < stop:
            try:
                r = requests.get(ARCTIC, params={
                    "subreddit": sub, "after": after, "before": stop,
                    "limit": PAGE, "sort": "asc",
                }, headers=UA, timeout=60)
                r.raise_for_status()
                data = r.json().get("data", [])
            except Exception:
                break
            if not data:
                break
            rows.extend(data)
            got += len(data)
            after = int(data[-1].get("created_utc", after)) + 1
            time.sleep(0.25)
        start = stop
    return rows


def main():
    print("Collecting real 2026 World Cup Reddit comments from Arctic Shift...")
    raw = []
    per_sub = {}
    for sub in SUBREDDITS:
        rows = pull_subreddit(sub)
        per_sub[sub] = len(rows)
        raw.extend((sub, c) for c in rows)
        print(f"  r/{sub}: {len(rows)} comments")

    # id -> author, to resolve reply edges to a fan
    id_author = {}
    for sub, c in raw:
        cid = c.get("id")
        a = c.get("author")
        if cid and a and a != "[deleted]":
            id_author[f"t1_{cid}"] = a

    events = []
    for sub, c in raw:
        a = c.get("author")
        if not a or a == "[deleted]":
            continue
        t = (float(c.get("created_utc", ORIGIN)) - ORIGIN) / 86400.0
        parent = c.get("parent_id", "") or ""
        reply_to = id_author.get(parent) if parent.startswith("t1_") else None
        if reply_to == a:
            reply_to = None
        events.append({
            "fan": f"u/{a}", "community": f"r/{sub}", "t": t,
            "reply_to": (f"u/{reply_to}" if reply_to else None), "platform": "reddit",
        })

    fans = {e["fan"] for e in events}
    comms = {e["community"] for e in events}
    replies = sum(1 for e in events if e["reply_to"])

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "wc2026_reddit_events.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    provenance = {
        "source": "Arctic Shift archive (Reddit historical, no auth)",
        "subreddits": SUBREDDITS, "per_subreddit_comments": per_sub,
        "window_utc": ["2026-06-11", "2026-07-20"],
        "retrieved": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "comments": len(events), "fans": len(fans), "communities": len(comms),
        "reply_edges": replies, "note": "Reddit only; Bluesky live window not captured.",
    }
    with open(data_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nREAL sample: {len(events)} comments | {len(fans)} fans | "
          f"{len(comms)} communities | {replies} reply edges")
    print(f"saved to data/wc2026_reddit_events.jsonl (+ provenance.json)")
    return events, provenance


if __name__ == "__main__":
    main()
