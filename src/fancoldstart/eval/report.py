"""Assemble hypothesis-test results into a human-readable report, with an
explicit provenance banner so a reader always knows whether the numbers came
from real collected data or from the synthetic pipeline-validation generator.
"""


def to_markdown(results, params, provenance, n_fans, n_cold, tau):
    banner = {
        "synthetic": (
            "SYNTHETIC PIPELINE-VALIDATION RUN. These numbers come from labeled "
            "synthetic data used only to confirm the pipeline executes end to end. "
            "They are not research results and appear nowhere in the paper."
        ),
        "real": (
            "REAL-DATA RUN. These numbers are pipeline outputs on collected public "
            "data. They are produced by running the released pipeline, as the paper "
            "states, and were not asserted in advance."
        ),
    }.get(provenance, f"DATA PROVENANCE: {provenance}.")

    lines = []
    lines.append("# Fan cold-start pipeline report")
    lines.append("")
    lines.append(f"> {banner}")
    lines.append("")
    lines.append(f"- Fans in calibration: {n_fans}")
    lines.append(f"- Cold-start fans (x <= threshold): {n_cold}")
    lines.append(f"- Future horizon tau: {tau}")
    lines.append(f"- BG/NBD population fit: r={params['r']:.4f}, alpha={params['alpha']:.4f}, "
                 f"a={params['a']:.4f}, b={params['b']:.4f}")
    lines.append("")
    lines.append("| Hypothesis | Family | Effect | value | p | p (BH) | SESOI met | Supported |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        fam = "primary" if r["primary"] else "secondary"
        pbh = "n/a" if r["p_bh"] is None else f"{r['p_bh']:.4f}"
        lines.append(
            f"| {r['name']} | {fam} | {r['effect_name']} | {r['effect']:.4f} | "
            f"{r['p']:.4f} | {pbh} | {r['sesoi_met']} | {r['supported']} |"
        )
    lines.append("")
    lines.append("Notes:")
    for r in results:
        if r["note"]:
            lines.append(f"- {r['name']}: {r['note']}")
    lines.append("")
    lines.append("Support decision for a primary hypothesis requires the correct sign, "
                 "the pre-registered smallest effect size of interest, and significance "
                 "surviving Benjamini-Hochberg at FDR 0.05.")
    return "\n".join(lines)
