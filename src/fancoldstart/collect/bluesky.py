"""Bluesky collector via the AT Protocol Jetstream firehose (free, no key).

Jetstream is a public WebSocket stream of all public AT Protocol events. This
module subscribes, filters posts to football and World Cup keywords, and
normalizes them to the pipeline event schema. Bluesky is a complementary live
source of posts and the follow graph; the formal fan graph comes from Reddit.

websockets is imported lazily so the core pipeline and tests do not require it.
"""
import json
import time

JETSTREAM_INSTANCES = [
    "wss://jetstream1.us-east.bsky.network/subscribe",
    "wss://jetstream2.us-east.bsky.network/subscribe",
    "wss://jetstream1.us-west.bsky.network/subscribe",
    "wss://jetstream2.us-west.bsky.network/subscribe",
]

DEFAULT_KEYWORDS = [
    "world cup", "worldcup", "fifa", "#wc2026", "usmnt", "football", "soccer",
]


def _matches(text, keywords):
    t = (text or "").lower()
    return any(k in t for k in keywords)


def collect_stream(origin_epoch, duration_s=300, keywords=None,
                   instance=JETSTREAM_INSTANCES[0]):
    """Subscribe to Jetstream for duration_s seconds, returning football/World Cup
    posts as events. Community is set to 'bsky' since Bluesky has no subreddit
    equivalent; the reply graph is captured from post reply references."""
    try:
        from websockets.sync.client import connect  # lazy
    except ImportError as e:
        raise RuntimeError(
            "websockets is required for Bluesky collection: pip install websockets"
        ) from e

    keywords = keywords or DEFAULT_KEYWORDS
    events = []
    deadline = None
    url = instance + "?wantedCollections=app.bsky.feed.post"
    with connect(url) as ws:
        while True:
            if deadline is None:
                deadline = time.monotonic() + duration_s
            if time.monotonic() > deadline:
                break
            try:
                msg = ws.recv(timeout=5)
            except TimeoutError:
                continue
            try:
                evt = json.loads(msg)
            except json.JSONDecodeError:
                continue
            commit = evt.get("commit") or {}
            record = commit.get("record") or {}
            if record.get("$type") != "app.bsky.feed.post":
                continue
            text = record.get("text", "")
            if not _matches(text, keywords):
                continue
            did = evt.get("did")
            created = record.get("createdAt", "")
            reply = (record.get("reply") or {}).get("parent", {}).get("uri")
            reply_did = None
            if reply and reply.startswith("at://"):
                reply_did = reply.split("/")[2]
            events.append({
                "fan": did, "community": "bsky",
                "t": _iso_to_days(created, origin_epoch),
                "reply_to": reply_did, "platform": "bluesky",
            })
    return events


def _iso_to_days(iso, origin_epoch):
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (dt.timestamp() - float(origin_epoch)) / 86400.0
    except (ValueError, AttributeError):
        return 0.0
