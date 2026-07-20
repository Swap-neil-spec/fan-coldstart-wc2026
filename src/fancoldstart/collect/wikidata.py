"""Wikidata / Wikipedia collector (CC0 / CC BY-SA 4.0).

Wikidata supplies the structural spine of the study: match results, venues, and
attendance. This module runs SPARQL against the public endpoint (no key) and
returns rows. The exact tournament-match query depends on how the 2026 event is
modeled in Wikidata at collection time, so the query string lives in config and
is passed in; a documented default is provided. Attendance figures obtained here
are the only in-tournament aggregate numbers the paper states, and they are
verified public record, not model outputs.
"""
import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "fan-coldstart-wc2026/0.1 (research; contact via repository)"

# A starting-point query for 2026 FIFA World Cup matches with attendance.
# Q108671319 is the 2026 FIFA World Cup item; verify the property path against the
# live model before relying on results (documented in DATA_LICENSES.md).
DEFAULT_MATCHES_QUERY = """
SELECT ?match ?matchLabel ?date ?attendance ?venueLabel WHERE {
  ?match wdt:P31/wdt:P279* wd:Q16466010 .      # instance of association football match
  ?match wdt:P361 wd:Q108671319 .              # part of the 2026 FIFA World Cup
  OPTIONAL { ?match wdt:P585 ?date . }
  OPTIONAL { ?match wdt:P1101 ?attendance . }  # number of spectators
  OPTIONAL { ?match wdt:P276 ?venue . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?date
"""


def run_sparql(query, endpoint=SPARQL_ENDPOINT, timeout=60):
    """Execute a SPARQL query and return a list of dict rows (values only)."""
    resp = requests.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for b in data["results"]["bindings"]:
        rows.append({k: v.get("value") for k, v in b.items()})
    return rows


def fetch_matches(query=DEFAULT_MATCHES_QUERY):
    """Return match rows (match, label, date, attendance, venue) for the tournament."""
    return run_sparql(query)


def total_and_peak_attendance(matches):
    """Aggregate cumulative and single-day peak attendance from match rows. Only
    matches carrying a spectator count contribute; the coverage is reported so a
    partial record is never presented as complete."""
    from collections import defaultdict

    by_day = defaultdict(float)
    total = 0.0
    counted = 0
    for m in matches:
        att = m.get("attendance")
        if att is None:
            continue
        val = float(att)
        total += val
        counted += 1
        day = (m.get("date") or "")[:10]
        by_day[day] += val
    peak_day, peak_val = (None, 0.0)
    for day, val in by_day.items():
        if val > peak_val:
            peak_day, peak_val = day, val
    return {
        "cumulative": total,
        "matches_with_attendance": counted,
        "matches_total": len(matches),
        "single_day_peak": peak_val,
        "single_day_peak_date": peak_day,
    }
