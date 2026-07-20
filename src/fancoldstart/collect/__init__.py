"""Collectors for real public sources (reddit, bluesky, wikidata) and a labeled
synthetic generator used only for pipeline validation. The real collectors are
imported directly by callers so that their optional third-party dependencies
(praw, atproto, websockets) are not required to run the core pipeline or tests.
"""
from . import synthetic  # noqa: F401
