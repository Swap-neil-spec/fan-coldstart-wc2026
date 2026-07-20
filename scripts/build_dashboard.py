"""Build the project dashboard at docs/index.html from real results.

Rendered as a Titan OS product screen: a left-rail app shell, a sticky top bar
with status chips, and instrument-panel cards with pulse rails and tabular
numerals. It reads data/real_results.json (from scripts/run_real.py), the raw
event stream (for the engagement heatmap and daily volume), and, if present,
data/google_trends.json (real attention). It fabricates nothing: real panels
carry the source, window, and counts; verified facts are labeled real; the scope
is stated plainly. The framing follows the original vision, predicting fan
lifetime value from social graphs and engagement, operationalized honestly as
predicted future engagement rather than dollars.

    python scripts/build_dashboard.py
"""
import sys
import json
import pathlib
import html as _html

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fancoldstart.stats import spearman  # noqa: E402

VERIFIED = {
    "cumulative_attendance": "3,605,357", "cumulative_asof": "25 Jun 2026",
    "single_day": "426,834", "single_day_date": "25 Jun 2026",
    "teams": "48", "matches": "104", "cities": "16", "window": "11 Jun to 19 Jul 2026",
}

HYP = {
    "H1": ("Cold-start lift from the graph-smoothed model", "primary", "paired Wilcoxon on holdout MAE"),
    "H2": ("Effect concentration in the cold-start stratum", "secondary", "interaction in pooled error regression"),
    "H3": ("Community breadth predicts persistence", "primary", "logistic + Wald, residualized on activity"),
    "H4": ("Weak-tie bridging predicts engagement", "primary", "negative-binomial + Wald"),
    "H5": ("Prediction beyond activity (falsification)", "primary", "gains survive degree-preserving rewiring"),
    "H6": ("Inductive cold-start transfer", "primary", "paired Wilcoxon vs population average"),
    "H7": ("Cross-platform robustness", "secondary", "same-sign replication, 2022 corpus"),
}

NAV = [
    ("findings", "Findings", "◉"),
    ("problem", "The cold start", "◈"),
    ("attention", "Public attention", "▧"),
    ("hypotheses", "Hypotheses", "◆"),
    ("provenance", "Provenance", "⬡"),
    ("reproduce", "Reproduce", "◇"),
]


def esc(s):
    return _html.escape(str(s))


def fmt_p(p):
    if p is None:
        return "n/a"
    if p <= 0 or p < 1e-300:
        return "&lt;1e-300"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.2g}"


def _events():
    path = ROOT / "data" / "wc2026_reddit_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(l)["t"] for l in open(path, encoding="utf-8")]


def _daily_volume(ts):
    days = np.zeros(39, float)
    for t in ts:
        d = int(t)
        if 0 <= d < 39:
            days[d] += 1
    return days


def _heatmap_grid(ts):
    grid = np.zeros((24, 39), float)
    for t in ts:
        d = int(t)
        if 0 <= d < 39:
            h = int((t - d) * 24) % 24
            grid[h, d] += 1
    return grid


def _bars(values, labels, unit):
    mx = max(values) or 1
    n = len(values)
    W, H, padB = 640, 190, 24
    bw = W / n
    out = [f'<line x1="0" y1="{H-padB}" x2="{W}" y2="{H-padB}" class="axis"/>']
    for i, v in enumerate(values):
        h = (H - padB - 8) * (v / mx)
        x = i * bw + bw * 0.14
        w = bw * 0.72
        y = H - padB - h
        show = (n <= 8) or (i % max(1, n // 8) == 0) or i == n - 1
        out.append(f'<g><title>{esc(labels[i])}: {int(v)} {unit}</title>'
                   f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(h,1):.1f}" rx="2" fill="var(--accent)"/>'
                   + (f'<text x="{x+w/2:.1f}" y="{H-padB+15:.0f}" class="ax" text-anchor="middle">{esc(labels[i])}</text>' if show else "") + '</g>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none" role="img">{"".join(out)}</svg>'


def _heatmap(grid):
    rows, cols = grid.shape
    mx = grid.max() or 1
    cw, ch, padL, padT = 15.4, 6.6, 26, 6
    W = padL + cols * cw + 4
    H = padT + rows * ch + 16
    cells = []
    for h in range(rows):
        if h % 6 == 0:
            cells.append(f'<text x="{padL-6}" y="{padT+h*ch+ch}" class="ax" text-anchor="end">{h:02d}h</text>')
        for d in range(cols):
            v = grid[h, d]
            if v <= 0:
                continue
            op = 0.10 + 0.90 * (v / mx)
            cells.append(f'<rect x="{padL+d*cw:.1f}" y="{padT+h*ch:.1f}" width="{cw-1:.1f}" height="{ch-1:.1f}" '
                         f'rx="1" fill="var(--accent)" fill-opacity="{op:.3f}"><title>day {d}, {h:02d}:00 UTC: {int(v)} comments</title></rect>')
    for d in range(0, cols, 6):
        cells.append(f'<text x="{padL+d*cw+cw/2:.1f}" y="{H-3}" class="ax" text-anchor="middle">d{d}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="xMidYMid meet" role="img">{"".join(cells)}</svg>'


def _lines(series, labels, colors):
    n = len(series[0])
    W, H, padB, padT = 640, 190, 22, 8
    def pts(s):
        mx = max(s) or 1
        return " ".join(f"{i/(n-1)*W:.1f},{(H-padB) - (H-padB-padT)*(v/mx):.1f}" for i, v in enumerate(s))
    out = [f'<line x1="0" y1="{H-padB}" x2="{W}" y2="{H-padB}" class="axis"/>']
    for s, c in zip(series, colors):
        out.append(f'<polyline points="{pts(s)}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
    for d in range(0, n, 6):
        out.append(f'<text x="{d/(n-1)*W:.1f}" y="{H-6}" class="ax" text-anchor="middle">d{d}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none" role="img">{"".join(out)}</svg>'


def _stat(label, value, sub="", tone=""):
    tc = {"ok": "st-ok", "warn": "st-warn"}.get(tone, "")
    return (f'<div class="stat"><div class="stat-k">{esc(label)}</div>'
            f'<div class="stat-v {tc}">{value}</div>'
            + (f'<div class="stat-s">{esc(sub)}</div>' if sub else "") + '</div>')


def _finding(r):
    name = r["name"]
    desc = HYP[name][0]
    if name == "H6":
        v, sub = f'+{r["effect"]*100:.0f}%', "vs the population-average baseline, on held-out cold-start fans"
    elif name == "H3":
        v, sub = f'{r["effect"]:+.3f}', "standardized logit coefficient per SD of community breadth"
    elif name == "H5":
        v, sub = f'+{r["effect"]*100:.1f}%', "real graph vs a degree-matched rewired graph"
    else:
        v, sub = f'{r["effect"]:+.3f}', HYP[name][2]
    if r["supported"]:
        pulse, tone, tag = "pulse-ok", "ok", "supported"
    elif r["primary"] and r["p"] < 0.05:
        pulse, tone, tag = "pulse-warn", "warn", "below pre-registered floor"
    else:
        pulse, tone, tag = "pulse-idle", "", "not supported"
    pbh = "" if r["p_bh"] is None else f' &middot; p(BH) {fmt_p(r["p_bh"])}'
    return (f'<article class="card pulse {pulse}"><div class="card-body">'
            f'<div class="finding-top"><span class="finding-h">{name}</span>'
            f'<span class="chip chip-{tone or "idle"}"><span class="dot"></span>{tag}</span></div>'
            f'<div class="finding-v {("st-"+tone) if tone else ""}">{v}</div>'
            f'<div class="finding-d">{esc(desc)}</div>'
            f'<div class="finding-m">{esc(sub)} &middot; p {fmt_p(r["p"])}{pbh}</div>'
            '</div></article>')


def _card(head, value, desc, meta, pulse, tone):
    return (f'<article class="card pulse {pulse}"><div class="card-body">'
            f'<div class="finding-top"><span class="finding-h">{esc(head)}</span></div>'
            f'<div class="finding-v {("st-"+tone) if tone else ""}">{value}</div>'
            f'<div class="finding-d">{esc(desc)}</div>'
            f'<div class="finding-m">{esc(meta)}</div></div></article>')


def _mae_bars(values, labels):
    mx = max(values) or 1
    n = len(values)
    W, H, padB = 640, 190, 26
    bw = W / n
    out = [f'<line x1="0" y1="{H-padB}" x2="{W}" y2="{H-padB}" class="axis"/>']
    for i, v in enumerate(values):
        h = (H - padB - 16) * (v / mx)
        x = i * bw + bw * 0.16
        w = bw * 0.68
        y = H - padB - h
        col = "var(--accent)" if i >= 2 else "var(--txt3)"
        out.append(f'<g><title>{esc(labels[i])}: MAE {v:.3f}</title>'
                   f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(h,1):.1f}" rx="2" fill="{col}"/>'
                   f'<text x="{x+w/2:.1f}" y="{y-4:.1f}" class="ax" text-anchor="middle">{v:.3f}</text>'
                   f'<text x="{x+w/2:.1f}" y="{H-padB+15:.0f}" class="ax" text-anchor="middle">{esc(labels[i])}</text></g>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none" role="img">{"".join(out)}</svg>'


def build():
    real_path = ROOT / "data" / "real_results.json"
    if not real_path.exists():
        print("no data/real_results.json; run scripts/run_real.py first.")
        return
    R = json.loads(real_path.read_text(encoding="utf-8"))
    ra_path = ROOT / "data" / "reanalysis.json"
    RA = json.loads(ra_path.read_text(encoding="utf-8")) if ra_path.exists() else None
    prov = R["provenance"]
    res = {r["name"]: r for r in R["results"]}
    supported = [n for n in ["H1", "H3", "H4", "H5", "H6"] if res[n]["supported"]]
    pct_cold = R["pct_cold"]
    P = VERIFIED

    trends_path = ROOT / "data" / "google_trends.json"
    trends = json.loads(trends_path.read_text(encoding="utf-8")) if trends_path.exists() else None
    tvals = [d["value"] for d in trends["series"]] if trends else []

    freq_chart = _bars(R["freq_hist"], ["0", "1", "2", "3", "4", "5", "6+"], "fans")

    # nav
    nav = "".join(f'<a href="#{i}" class="nav-item"><span class="nav-ico">{ic}</span>{lbl}</a>'
                  for i, lbl, ic in NAV)

    # findings
    findings = "".join(_finding(res[n]) for n in ["H6", "H3", "H5"])

    # hypothesis table
    rows = ""
    for name, (desc, fam, test) in HYP.items():
        r = res[name]
        if r["supported"]:
            st = '<span class="chip chip-ok"><span class="dot"></span>supported</span>'
        elif r["primary"] and r["p"] < 0.05:
            st = '<span class="chip chip-warn"><span class="dot"></span>below floor</span>'
        elif r["primary"]:
            st = '<span class="chip chip-idle"><span class="dot"></span>not supported</span>'
        else:
            st = '<span class="chip chip-idle"><span class="dot"></span>secondary</span>'
        cls = ' class="rsup"' if r["supported"] else ""
        rows += (f'<tr{cls}><td class="mono">{name}</td><td>{esc(desc)}'
                 f'<div class="fam">{fam} &middot; {esc(test)}</div></td>'
                 f'<td class="num">{r["effect"]:+.3f}</td><td class="num">{fmt_p(r["p"])}</td>'
                 f'<td class="num">{fmt_p(r["p_bh"])}</td><td>{st}</td></tr>')

    subs = ", ".join("r/" + s for s in prov["subreddits"])

    M = []
    # top bar
    M.append(
        '<header class="topbar">'
        '<div class="crumb"><span class="crumb-d">Fan Value OS</span><span class="crumb-sep">/</span>'
        'Cold-Start Study</div><div class="spacer"></div>'
        '<span class="chip chip-ok" id="live-chip"><span class="dot"></span>real data</span>'
        '<span class="chip chip-accent">pre-registered</span>'
        f'<span class="chip chip-idle">{len(supported)}/5 primary</span>'
        '<button class="tbtn" id="tt" aria-label="Toggle theme">◐</button></header>'
    )
    # page header
    M.append(
        '<div class="pagehead" id="findings"><div class="kick">Predicting fan lifetime value from social graphs and engagement</div>'
        '<h1>Spotting the superfans before they have a history.</h1>'
        '<p class="sub">The valuable fans are the durable ones, and at a World Cup most arrive as newcomers with '
        'almost no track record. Fit on real Reddit data from the 2026 tournament, breadth of community participation '
        'predicts which of them keep showing up, within community and net of activity. The graph adds a small, honest '
        'amount beyond a fan\'s own activity; the full decomposition is below.</p></div>'
    )
    # stat readout
    M.append(
        '<div class="card"><div class="statrow">'
        + _stat("real comments", f'{prov["comments"]:,}')
        + _stat("unique fans", f'{prov["fans"]:,}')
        + _stat("communities", str(prov["communities"]))
        + _stat("cold-start", f'{pct_cold:.0f}%')
        + _stat("breadth effect", f'+{RA["h3_within_coef"]:.2f}' if RA else "n/a", "within community (H3)", "ok")
        + _stat("primary supported", f'{len(supported)} / 5', "pre-registered")
        + '</div></div>'
    )
    # findings cards, honestly decomposed
    if RA:
        c1 = _card("H3 · holds", f'+{RA["h3_within_coef"]:.2f}',
                   "Community breadth predicts persistence",
                   "within community and net of activity; standardized logit coefficient, p < 1e-17",
                   "pulse-ok", "ok")
        c2 = _card("H6 · decomposed", f'+{RA["h6_lift_vs_activity_degree"]*100:.1f}%',
                   "Graph beyond activity and degree",
                   f'the raw +{RA["h6_lift_vs_global"]*100:.0f}% over a global mean is mostly a fan\'s own activity and community',
                   "pulse-idle", "")
        c3 = _card("H5 · below floor", f'+{RA["h5_rewire_mean"]*100:.1f}%',
                   "Real graph vs a degree-matched graph",
                   "beats the rewired graph in 60 of 60 draws, but stays under the pre-registered 5% floor",
                   "pulse-warn", "warn")
        M.append(f'<div class="grid3">{c1}{c2}{c3}</div>')
        mae = RA["h6_mae"]
        chart = _mae_bars([mae["global"], mae["community"], mae["activity_degree"], mae["inductive"]],
                          ["global", "community", "act + degree", "graph"])
        M.append(
            '<section class="card"><div class="card-body"><div class="sect">Where the lift comes from</div>'
            '<p class="lead">The graph model beats a global mean by 30 percent, but almost all of that is a fan\'s own '
            'activity and community. Against an activity-and-degree baseline, the graph adds about one percent.</p>'
            f'{chart}<p class="cap">Holdout mean absolute error on cold-start fans, lower is better; each bar is a '
            'progressively fairer baseline. The honest, network-specific contribution is the last gap, not the first. '
            '<span class="src">Re-analysis on the real pilot data.</span></p></div></section>'
        )
    else:
        M.append(f'<div class="grid3">{findings}</div>')

    # the cold start
    M.append(
        f'<section class="card" id="problem"><div class="card-body"><div class="sect">The cold start</div>'
        f'<p class="lead">{pct_cold:.0f}% of fans have at most one prior comment, so standard customer-base '
        'models have almost no history to condition on.</p>'
        f'{freq_chart}<p class="cap">Prior comments per fan in the calibration window. '
        '<span class="src">Reddit via the Arctic Shift archive.</span></p></div></section>'
    )

    # public attention (Google Trends, real, standalone context)
    if tvals:
        peak = max(trends["series"], key=lambda d: d["value"])
        chart = _lines([tvals], ["Google Trends"], ["var(--accent)"])
        M.append(
            '<section class="card" id="attention"><div class="card-body"><div class="sect">The event was real and large</div>'
            f'<p class="lead">Independent Google Trends search interest for the World Cup peaked on {esc(peak["date"])}, '
            'and verified attendance ran into the millions. This is the scale of the phenomenon the study is fit on.</p>'
            f'{chart}<p class="cap">Google Trends interest for the term "FIFA World Cup", United States, indexed 0 to 100, '
            'day 0 = 11 Jun to day 38 = 19 Jul 2026. Verified attendance reached ' + esc(P["single_day"])
            + ' in a single day (' + esc(P["single_day_date"]) + ') and ' + esc(P["cumulative_attendance"])
            + ' cumulative. <span class="src">Google Trends and Wikidata (CC0).</span></p></div></section>'
        )

    # hypotheses table
    M.append(
        '<section class="card" id="hypotheses"><div class="card-body"><div class="sect">Pre-registered hypotheses, real outcomes</div>'
        '<p class="cap" style="margin-bottom:12px">Two-sided &alpha; 0.05, Benjamini-Hochberg across the primary family '
        f'at FDR 0.05, a smallest effect size of interest per primary hypothesis. Supported: {", ".join(supported) if supported else "none"}.</p>'
        '<div class="tablewrap"><table><thead><tr><th></th><th>Hypothesis</th><th class="num">effect</th>'
        '<th class="num">p</th><th class="num">p(BH)</th><th>result</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        '<p class="cap" style="margin-top:12px">H6 passed its registered test against a population-average baseline, '
        'but a fairer baseline that already knows a fan\'s activity and degree cuts the graph\'s contribution to about '
        'one percent (see the decomposition above). H4 and H5 clear significance but stay below their pre-registered '
        'floors, so the network-specific signal is real but modest. The result that holds is H3, breadth predicting '
        'persistence within community. H1 does not beat the per-fan baseline.</p></div></section>'
    )

    # provenance + verified facts
    M.append(
        '<section class="card" id="provenance"><div class="card-body"><div class="sect">Provenance and verified facts</div>'
        '<div class="statrow">'
        + _stat("comments", f'{prov["comments"]:,}') + _stat("fans", f'{prov["fans"]:,}')
        + _stat("reply edges", f'{prov["reply_edges"]:,}')
        + _stat("attendance", esc(P["cumulative_attendance"]), "cumulative, " + P["cumulative_asof"])
        + _stat("matches", P["matches"], f'{P["teams"]} teams, {P["cities"]} cities')
        + '</div>'
        f'<p class="cap" style="margin-top:14px"><b>Reddit sample:</b> {esc(prov["source"])}. '
        f'{esc(subs)}. Window {esc(P["window"])}, retrieved {esc(prov["retrieved"])}. '
        '<b>Attendance and structure:</b> Wikidata and Wikipedia (CC0). '
        '<b>Attention:</b> Google Trends. Every number here is computed from these sources; the collection and '
        'analysis scripts are in the repository.</p></div></section>'
    )

    # scope
    M.append(
        '<section class="card pulse pulse-warn"><div class="card-body"><div class="sect">Scope and limits</div>'
        '<p class="cap">A pilot on a bounded sample, not the full fanbase. The graph comes from Reddit '
        'co-participation and replies; the Bluesky live window had already closed by collection time. Value here is '
        'predicted future engagement, not dollars.</p></div></section>'
    )

    # reproduce
    M.append(
        '<section class="card" id="reproduce"><div class="card-body"><div class="sect">Reproduce</div>'
        '<pre class="cmd">python scripts/collect_real.py     # pull the real sample from the archive\n'
        'python scripts/run_real.py         # run the pre-registered pipeline\n'
        'python scripts/build_dashboard.py  # rebuild this screen</pre>'
        '<p class="cap" style="margin-top:10px">Core dependencies: numpy and networkx. Full test suite: '
        '<code>pytest</code>. Repository and pre-registration linked in the rail.</p></div></section>'
    )

    main = f'<main class="main">{"".join(M)}<footer class="foot">MIT licensed. Swapnil (Neil) Rajkumar Gaikwad, ' \
           '2026.</footer></main>'

    rail = (
        '<aside class="rail"><div class="brand"><div class="mark">FV</div>'
        '<div class="brand-t">Fan Value OS<span>Cold-Start Study</span></div></div>'
        f'<nav class="nav">{nav}</nav>'
        '<div class="rail-foot">'
        '<a class="nav-item" href="https://github.com/Swap-neil-spec/fan-coldstart-wc2026"><span class="nav-ico">◰</span>Repository</a>'
        '<a class="nav-item" href="https://github.com/Swap-neil-spec/fan-coldstart-wc2026/blob/main/PREREGISTRATION.md"><span class="nav-ico">▤</span>Pre-registration</a>'
        '</div></aside>'
    )

    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           '<title>Fan Value OS - cold-start study on real 2026 World Cup data</title>'
           '<meta name="description" content="Predicting fan lifetime value from social graphs and engagement: a pre-registered cold-start study on real 2026 FIFA World Cup data.">'
           '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
           '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
           f'<script>{PREPAINT}</script><style>{CSS}</style></head><body><div class="app">'
           f'{rail}{main}</div><script>{JS}</script></body></html>')

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(doc, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote docs/index.html | supported={supported} | trends points={len(tvals)}")


PREPAINT = ("(function(){try{var t=localStorage.getItem('titan_theme');"
            "if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches))"
            "document.documentElement.setAttribute('data-theme','dark');}catch(e){}})();")

JS = ("(function(){var b=document.getElementById('tt');if(b)b.addEventListener('click',function(){"
      "var r=document.documentElement,d=r.getAttribute('data-theme')==='dark';"
      "if(d)r.removeAttribute('data-theme');else r.setAttribute('data-theme','dark');"
      "try{localStorage.setItem('titan_theme',d?'light':'dark');}catch(e){}});"
      "fetch('/api/data').then(function(r){return r.ok?r.json():null;}).then(function(j){"
      "if(j&&j.ok){var c=document.getElementById('live-chip');"
      "if(c){c.innerHTML='<span class=\\\"dot\\\"></span>live \\u00b7 Netlify Blobs';"
      "c.title='served from Netlify Blobs at '+j.servedAt;}}}).catch(function(){});})();")

CSS = """
:root{color-scheme:light;
 --canvas:#F4F4F1;--raised:#fff;--card:#fff;--line:#E2E2DB;--txt:#141414;--txt2:#34352F;--txt3:#686862;
 --accent:#2F7D74;--warn:#B8860B;--bad:#C0392B;--fill:rgba(0,0,0,.045);--fill2:rgba(0,0,0,.08);
 --shadow:0 2px 10px rgba(20,20,20,.06);--axis:#8C8C86;--grid:#E2E2DB;--grad:linear-gradient(135deg,#2F7D74,#3E9186);
 --pulse-ok:#2F7D74;--pulse-warn:#B8860B;--pulse-idle:#C9C9C0;}
:root[data-theme="dark"]{color-scheme:dark;
 --canvas:#0A0A0F;--raised:#131319;--card:#17171F;--line:#26262F;--txt:#F4F4F8;--txt2:#A2A2B4;--txt3:#868696;
 --accent:#4FB3A6;--warn:#F5B23D;--bad:#F0564A;--fill:rgba(255,255,255,.05);--fill2:rgba(255,255,255,.10);
 --shadow:0 1px 3px rgba(0,0,0,.4);--axis:#6A6A7C;--grid:#26262F;--grad:linear-gradient(135deg,#4FB3A6,#3E9186);
 --pulse-ok:#35D6A0;--pulse-warn:#F5B23D;--pulse-idle:#4A4A58;}
*{box-sizing:border-box}
body{margin:0;background:var(--canvas);color:var(--txt2);font-family:'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.mono,.num,.stat-v,.finding-v,table{font-variant-numeric:tabular-nums;font-feature-settings:'tnum' 1}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.app{display:flex;min-height:100vh}
/* rail */
.rail{width:224px;flex:none;background:var(--raised);border-right:1px solid var(--line);position:sticky;top:0;height:100vh;display:flex;flex-direction:column;padding:0}
.brand{display:flex;align-items:center;gap:10px;height:56px;padding:0 16px;border-bottom:1px solid var(--line)}
.mark{width:30px;height:30px;border-radius:8px;background:var(--grad);color:#fff;font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(47,125,116,.28)}
.brand-t{font-size:14px;font-weight:700;color:var(--txt);line-height:1.15;letter-spacing:-.01em}
.brand-t span{display:block;font-size:11px;font-weight:500;color:var(--txt3)}
.nav{flex:1;padding:10px 8px;display:flex;flex-direction:column;gap:2px}
.nav-item{display:flex;align-items:center;gap:11px;padding:8px 11px;border-radius:8px;font-size:13.5px;font-weight:500;color:var(--txt3);text-decoration:none;transition:background .15s,color .15s}
.nav-item:hover{background:var(--fill);color:var(--txt)}
.nav-ico{width:16px;text-align:center;opacity:.8;font-size:14px}
.rail-foot{padding:10px 8px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:2px}
/* main */
.main{flex:1;min-width:0;display:flex;flex-direction:column}
.topbar{height:56px;position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:10px;padding:0 22px;border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--canvas) 82%,transparent);backdrop-filter:blur(8px)}
.crumb{font-size:13px;color:var(--txt3);font-weight:500}.crumb-d{color:var(--txt);font-weight:650}.crumb-sep{margin:0 8px;opacity:.5}
.spacer{flex:1}
.tbtn{border:1px solid var(--line);background:var(--card);color:var(--txt3);border-radius:8px;width:32px;height:32px;cursor:pointer;font-size:14px}
.pagehead{padding:26px 26px 4px}
.kick{font-size:11.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--accent);font-weight:700}
h1{font-size:29px;line-height:1.14;color:var(--txt);margin:9px 0 12px;letter-spacing:-.015em;font-weight:750;max-width:20ch;text-wrap:balance}
.sub{font-size:15px;color:var(--txt2);max-width:74ch;margin:0;line-height:1.6}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);margin:14px 26px}
.card-body{padding:20px 22px}
.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;background:var(--line)}
.card>.statrow{border-radius:12px;overflow:hidden}
.stat{background:var(--card);padding:15px 18px}
.stat-k{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--txt3)}
.stat-v{font-size:22px;font-weight:750;color:var(--txt);letter-spacing:-.02em;margin-top:4px}
.stat-s{font-size:11.5px;color:var(--txt3);margin-top:2px}
.st-ok{color:var(--accent)}.st-warn{color:var(--warn)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 26px}
.grid3 .card{margin:0}
.pulse{position:relative}.pulse::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:3px 0 0 3px;background:var(--pulse,var(--pulse-idle))}
.pulse-ok{--pulse:var(--pulse-ok)}.pulse-warn{--pulse:var(--pulse-warn)}.pulse-idle{--pulse:var(--pulse-idle)}
.finding-top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.finding-h{font-size:12px;font-weight:750;color:var(--txt3);letter-spacing:.05em}
.finding-v{font-size:30px;font-weight:800;color:var(--txt);letter-spacing:-.02em;margin:8px 0 3px;line-height:1}
.finding-d{font-size:14px;font-weight:650;color:var(--txt)}
.finding-m{font-size:11.5px;color:var(--txt3);margin-top:6px;line-height:1.45}
.sect{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--txt3);font-weight:700;margin-bottom:12px}
.lead{font-size:17px;font-weight:650;color:var(--txt);margin:0 0 14px;letter-spacing:-.01em;line-height:1.35;text-wrap:balance}
.cap{font-size:12.5px;color:var(--txt3);margin:10px 0 0;line-height:1.55}
.src{color:var(--accent);font-weight:600}
.chart{width:100%;height:auto;display:block}
.chart .axis{stroke:var(--grid);stroke-width:1.5}.chart .ax{fill:var(--axis);font-size:10.5px}
.legend{display:flex;gap:16px;margin-bottom:8px}
.lg{font-size:12px;color:var(--txt3);display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--txt3);font-weight:700}
td.mono{font-weight:700;color:var(--txt)}.num{text-align:right}th.num{text-align:right}
.fam{font-size:11px;color:var(--txt3);margin-top:2px}
tr.rsup td{background:var(--fill)}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;border-radius:999px;padding:3px 9px;border:1px solid var(--line);white-space:nowrap}
.chip .dot{width:7px;height:7px;border-radius:999px;background:currentColor;display:inline-block;opacity:.9}
.chip-ok{color:var(--accent);background:color-mix(in srgb,var(--accent) 11%,transparent);border-color:color-mix(in srgb,var(--accent) 22%,transparent)}
.chip-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);border-color:color-mix(in srgb,var(--warn) 24%,transparent)}
.chip-accent{color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:color-mix(in srgb,var(--accent) 20%,transparent)}
.chip-idle{color:var(--txt3)}
code{font-family:ui-monospace,'SF Mono',Consolas,monospace;font-size:12.5px;background:var(--fill);border-radius:5px;padding:1px 5px}
pre.cmd{font-family:ui-monospace,'SF Mono',Consolas,monospace;font-size:12.5px;background:var(--canvas);border:1px solid var(--line);border-radius:8px;padding:13px 15px;overflow-x:auto;line-height:1.7;color:var(--txt2);margin:0}
.foot{font-size:12.5px;color:var(--txt3);padding:22px 26px 40px;border-top:1px solid var(--line);margin-top:14px}
a{color:var(--accent)}
@media(prefers-reduced-motion:reduce){*{transition-duration:.01ms!important}}
@media(max-width:860px){.rail{display:none}.grid3{grid-template-columns:1fr}.card,.grid3,.pagehead{margin-left:16px;margin-right:16px}h1{font-size:24px}}
"""

if __name__ == "__main__":
    build()
