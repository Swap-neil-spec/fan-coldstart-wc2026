"""Reddit collector: live via the official Data API (PRAW) and historical via
the Arctic Shift archive.

The official Reddit Data API free tier requires prior approval under the
November 2025 Responsible Builder policy and is restricted to non-commercial use,
which academic research satisfies. Set credentials in the environment:
REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT. praw is imported lazily
so the core pipeline and tests do not require it.

Every collected item is normalized to the pipeline event schema:
  {fan, community, t, reply_to, platform}
where t is days from `origin_epoch`. Reddit supplies the formal fan graph.
"""
import os
import time

import requests

ARCTIC_SHIFT = "https://arctic-shift.photon-reddit.com/api"


def _to_days(epoch, origin_epoch):
    return (float(epoch) - float(origin_epoch)) / 86400.0


def collect_live(subreddits, origin_epoch, limit_per_sub=1000):
    """Collect recent submissions and comments from the given subreddits via PRAW.
    Returns events in the pipeline schema. Requires Reddit API credentials."""
    try:
        import praw  # lazy: only needed for live collection
    except ImportError as e:
        raise RuntimeError("praw is required for live Reddit collection: pip install praw") from e

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "fan-coldstart-wc2026/0.1"),
    )
    events = []
    for sub in subreddits:
        sr = reddit.subreddit(sub)
        for submission in sr.new(limit=limit_per_sub):
            events.append({
                "fan": f"u/{submission.author.name}" if submission.author else "u/[deleted]",
                "community": f"r/{sub}", "t": _to_days(submission.created_utc, origin_epoch),
                "reply_to": None, "platform": "reddit",
            })
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                if not c.author:
                    continue
                parent = getattr(c, "parent_id", "")
                reply_to = None
                if parent.startswith("t1_"):
                    try:
                        pa = reddit.comment(parent[3:]).author
                        reply_to = f"u/{pa.name}" if pa else None
                    except Exception:
                        reply_to = None
                events.append({
                    "fan": f"u/{c.author.name}", "community": f"r/{sub}",
                    "t": _to_days(c.created_utc, origin_epoch),
                    "reply_to": reply_to, "platform": "reddit",
                })
    return events


def collect_historical(subreddit, after_epoch, before_epoch, origin_epoch,
                       kind="comments", limit=100, pause=1.0):
    """Collect historical items from the Arctic Shift archive (no auth) for
    pre-tournament cold-start baselines. Paginates by time."""
    events = []
    url = f"{ARCTIC_SHIFT}/{kind}/search"
    after = int(after_epoch)
    while after < int(before_epoch):
        resp = requests.get(url, params={
            "subreddit": subreddit, "after": after, "before": int(before_epoch),
            "limit": limit, "sort": "asc",
        }, headers={"User-Agent": "fan-coldstart-wc2026/0.1"}, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for item in data:
            author = item.get("author")
            if not author or author == "[deleted]":
                continue
            parent = item.get("parent_id", "") or ""
            events.append({
                "fan": f"u/{author}", "community": f"r/{subreddit}",
                "t": _to_days(item.get("created_utc", after), origin_epoch),
                "reply_to": None if not parent.startswith("t1_") else parent,
                "platform": "reddit",
            })
        after = int(data[-1].get("created_utc", after)) + 1
        time.sleep(pause)
    return events
