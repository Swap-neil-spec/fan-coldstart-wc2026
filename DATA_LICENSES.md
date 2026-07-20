# Data sources, licenses, and redistribution

No third-party data is bundled in this repository. The code collects data from
public sources at run time, each governed by its own terms. This file records
what may and may not be redistributed, so a reader knows exactly how to reproduce
collection and what the code can and cannot ship.

## Sources

- Wikidata (CC0). Match structure, venues, and attendance. Freely redistributable.
  This is the only source whose data may be committed to the repository, and the
  attendance figures used in the paper come from here.
- Wikipedia (CC BY-SA 4.0). Supplementary match and tournament facts. Shareable
  with attribution and share-alike.
- Reddit Data API (official). The formal fan graph. Free non-commercial tier under
  the November 2025 Responsible Builder policy, which requires prior approval.
  Reddit content may not be rehosted or redistributed in bulk, so the repository
  ships code that re-collects it, never the collected corpus.
- Bluesky / AT Protocol Jetstream (public firehose, no key). Complementary posts
  and follow graph. Real-time only: past events are not re-pullable, so a capture
  is a point-in-time snapshot. Redistribution is governed by AT Protocol and user
  terms; the repository ships the collector, not the captured posts.
- Arctic Shift (Reddit historical archive). Pre-tournament baselines. Same Reddit
  content terms apply; not redistributed here.
- 2022 Qatar World Cup Twitter dataset (Journal of Computational Social Science,
  DOI 10.1007/s42001-025-00410-x). Historical cross-platform supplement, used under
  its own terms; not redistributed here.
- StatsBomb Open Data. On-pitch event context for prior tournaments, under the
  StatsBomb user agreement and attribution requirement; not redistributed here.

## Reproducibility and its limits

The code is fully reproducible: the pipeline, the pre-registration, and the
Wikidata-derived facts can be re-run by anyone at no cost. The live data is not
fully re-creatable after the fact, because the Reddit tier is approval-gated and
non-redistributable and the Bluesky firehose carries only current events. This is
an honest limit of the free, public, live-data setting, and the paper states it.
Raw pulls should be stored locally under `data/` (git-ignored) with retrieval
timestamps so a given analysis is reproducible from the archived pull.

## Deliberately excluded

Paid live feeds (for example broadcast-grade event and tracking data, and paid
social APIs) are excluded so the entire live pipeline is reproducible at zero cost.
