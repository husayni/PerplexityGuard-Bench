"""Patch the REPORT.md TBD placeholders with numbers from stitch_attack_summary.csv.

Reads `perplexity_guard/audit/results/stitch_attack_summary.csv` (produced
by `stitch_attack.py`) and substitutes the per-cell detection rates into
the §4.5 table in `REPORT.md`. Also fills the abstract and §1
contribution-4 inline numbers. The narrative paragraph in §4.5
({STITCH_NARRATIVE}) is left to the author — it depends on what the data
shows. This script prints a recommended narrative skeleton with the
right numbers so the author can paste it in.

Usage:
    uv run python -m perplexity_guard.audit.patch_report
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "perplexity_guard" / "audit" / "results" / "stitch_attack_summary.csv"
REPORT = REPO_ROOT / "REPORT.md"

TARGET_FRACS = [0.0, 0.10, 0.25, 0.40, 0.50, 0.75, 1.0]
SCREENS = ["perplexity_only", "complexity_only", "or_gate"]
SCREEN_PRETTY = {
    "perplexity_only": "perplexity-only",
    "complexity_only": "complexity-only",
    "or_gate": "OR-gate",
}


def _detect_at(summary: pd.DataFrame, screen: str, frac: float) -> float | None:
    """Find the detection rate for a (screen, prefix_frac) cell, or None."""
    row = summary[(summary["screen"] == screen)
                  & (summary["prefix_frac"].round(3) == round(frac, 3))]
    if row.empty:
        return None
    return float(row["detect_rate"].iloc[0]) * 100


def main() -> int:
    if not SUMMARY.exists():
        print(f"[error] {SUMMARY} not found — run stitch_attack.py first.")
        return 2
    summary = pd.read_csv(SUMMARY)
    text = REPORT.read_text()

    # Build the per-cell substitution map for the §4.5 table.
    # Placeholder names in REPORT.md were normalised to TBD by the linter,
    # so we instead look for the table rows in the §4.5 section and replace
    # them in-place. The table is uniquely identified by the row "perplexity-only"
    # followed by " & TBD ..." cells.
    table_rows = []
    for screen in SCREENS:
        cells = []
        for f in TARGET_FRACS:
            v = _detect_at(summary, screen, f)
            cells.append(f"{v:.1f}" if v is not None else "—")
        table_rows.append((screen, cells))

    # Replace the table block in §4.5 with a freshly built one.
    new_table = (
        "\\small\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption*{\\textbf{Table 3: Detection rate (\\%) under stitching as a function of natural-prefix fraction}}\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "screen           & 0\\% & 10\\% & 25\\% & 40\\% & 50\\% & 75\\% & 100\\% \\\\\n"
        "\\midrule\n"
    )
    for screen, cells in table_rows:
        new_table += f"{SCREEN_PRETTY[screen]:<16}"
        for c in cells:
            new_table += f" & {c}"
        new_table += " \\\\\n"
    new_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\\normalsize\n"
    )

    # Replace the Table 3 block by matching from the small Table 3 begin to its end.
    table_re = re.compile(
        r"\\small\n\\begin\{table\}\[h\]\n\\centering\n"
        r"\\caption\*\{\\textbf\{Table 3:[^}]+\}\}\n"
        r"\\begin\{tabular\}\{[^}]+\}\n"
        r"\\toprule\n"
        r".*?"
        r"\\bottomrule\n\\end\{tabular\}\n\\end\{table\}\n\\normalsize\n",
        re.DOTALL,
    )
    # Use a lambda so re.sub does not interpret backslash escapes in the LaTeX
    # replacement (e.g. \small, \toprule) as regex escape sequences.
    new_text, n = table_re.subn(lambda _m: new_table, text, count=1)
    if n != 1:
        print("[warn] Table 3 substitution did not match — left unchanged.")
        new_text = text

    # Inline numbers in abstract + §1 contribution 4.
    perp_at_50 = _detect_at(summary, "perplexity_only", 0.50)
    perp_at_0 = _detect_at(summary, "perplexity_only", 0.00)
    or_at_50 = _detect_at(summary, "or_gate", 0.50)
    or_at_0 = _detect_at(summary, "or_gate", 0.00)

    # The current text has "TBD" placeholders left over from the linter swap.
    # We replace specific TBD usages by phrase pattern.

    # Abstract: "dropping perplexity-only detection to TBD at a TBD natural-prefix budget"
    new_text = re.sub(
        r"dropping perplexity-only detection to TBD at a TBD natural-prefix budget",
        f"dropping perplexity-only detection to {perp_at_50:.0f}% at a 50% natural-prefix budget",
        new_text,
    )

    # §1 contribution 4: "perplexity-only detection from TBD (full-adversary control) to TBD at a 50% natural-prefix budget; the OR-ensemble degrades from TBD to TBD"
    new_text = re.sub(
        r"perplexity-only detection from TBD \(full-adversary control\) to TBD at a 50% natural-prefix budget; the OR-ensemble degrades from TBD to TBD",
        f"perplexity-only detection from {perp_at_0:.0f}% (full-adversary control) to {perp_at_50:.0f}% at a 50% natural-prefix budget; the OR-ensemble degrades from {or_at_0:.0f}% to {or_at_50:.0f}%",
        new_text,
    )

    # Build a recommended narrative skeleton based on the data.
    # Find the smallest prefix_frac at which each screen drops below 50%.
    flip_points = {}
    for screen in SCREENS:
        sub = summary[summary["screen"] == screen].sort_values("prefix_frac")
        below = sub[sub["detect_rate"] < 0.5]
        if not below.empty:
            flip_points[screen] = float(below["prefix_frac"].iloc[0])
        else:
            flip_points[screen] = None

    narrative_lines = [
        "Three observations.",
        "",
        f"1. **The perplexity-only screen is structurally averaging-vulnerable.** "
        f"Detection drops monotonically with prefix length, from {perp_at_0:.0f}% on the full adversary "
        f"to {perp_at_50:.0f}% at 50% natural-prefix budget"
        + (f" and below 50% by prefix fraction {flip_points['perplexity_only']:.2f}."
           if flip_points["perplexity_only"] is not None else " (does not drop below 50% in tested range).")
        ,
        "",
        f"2. **The complexity-only screen is robust to stitching** — natural prefixes do not "
        f"introduce repetition, so the per-window entropy / k-mer signals are unaffected by the attack. "
        f"It does not catch the underlying mutated_40pct adversary either, however, so it cannot patch the gap.",
        "",
        f"3. **The OR-ensemble degrades to {or_at_50:.0f}% at 50% prefix**, confirming that the perplexity arm "
        f"dominates the OR-gate's behavior on this attack. Combining a vulnerable signal with a non-responsive "
        f"signal does not produce a robust ensemble against this attack.",
    ]

    print("\n=== Recommended §4.5 narrative paragraph (paste into REPORT.md, replacing 'TBD'-narrative placeholder) ===\n")
    print("\n".join(narrative_lines))
    print()
    print("=== Substitutions applied ===")
    print(f"  Table 3: {'replaced' if n == 1 else 'NOT replaced (leave a TBD warning)'}")
    print(f"  Abstract: perp@50 = {perp_at_50:.0f}%")
    print(f"  §1 contrib 4: perp 0->{perp_at_50:.0f}%, OR {or_at_0:.0f}->{or_at_50:.0f}%")

    REPORT.write_text(new_text)
    print(f"\nWrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
