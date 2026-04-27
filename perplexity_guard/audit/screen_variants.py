"""Three screen variants applied to the same per-sequence signal table.

The existing e2e_demo CSV stores raw per-sequence signals (perplexity,
z-score, repetitiveness, etc.) AND the OR-gate verdict. This module
re-derives the verdicts that each *individual* signal would have produced
on its own, so we can compare detection rates head-to-head.

This is what makes the audit actually an audit: the same sequences,
multiple decision rules, one matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


# Thresholds duplicated here so the audit module is self-contained and
# does not silently inherit changes to the main ensemble logic.
PPL_REVIEW_Z = 1.5
PPL_BLOCK_Z = 3.0
PPL_BORDERLINE_Z = 1.0     # used by OR-gate when commec is skipped

LOW_COMPLEXITY_REVIEW = 0.20
LOW_COMPLEXITY_BLOCK = 0.50


@dataclass
class ScreenVariant:
    name: str
    description: str
    decide: Callable[[pd.Series], str]


def perplexity_only(row: pd.Series) -> str:
    """ESM-2 pseudo-perplexity z-score gate, no other signal."""
    z = float(row["z_natural"])
    if z >= PPL_BLOCK_Z:
        return "BLOCK"
    if z >= PPL_REVIEW_Z or bool(row.get("is_high", False)):
        return "REVIEW"
    return "PASS"


def complexity_only(row: pd.Series) -> str:
    """Wootton-Federhen entropy + distinct-k-mer fraction, no perplexity."""
    rep = float(row["repetitiveness"])
    if rep >= LOW_COMPLEXITY_BLOCK:
        return "BLOCK"
    if rep >= LOW_COMPLEXITY_REVIEW:
        return "REVIEW"
    return "PASS"


def or_gate(row: pd.Series) -> str:
    """The OR over (perplexity z, complexity repetitiveness, commec).

    Reproduces the live ensemble logic so audit numbers match what the
    deployed PerplexityGuard would emit. Commec is treated as 'skipped'
    everywhere in this audit because the demo run did not have it
    configured; that means the OR-gate gets a borderline-z bump of
    REVIEW on z>=1.0 sequences.
    """
    z = float(row["z_natural"])
    rep = float(row["repetitiveness"])
    commec_skipped = str(row.get("commec_status", "skipped")) == "skipped"

    if rep >= LOW_COMPLEXITY_BLOCK:
        return "BLOCK"
    if z >= PPL_BLOCK_Z:
        return "BLOCK"
    if rep >= LOW_COMPLEXITY_REVIEW:
        return "REVIEW"
    if z >= PPL_REVIEW_Z or bool(row.get("is_high", False)):
        return "REVIEW"
    if commec_skipped and z >= PPL_BORDERLINE_Z:
        return "REVIEW"
    return "PASS"


def sliding_window_perplexity(row: pd.Series) -> str:
    """Position-resolved variant: BLOCK if any sliding window's z exceeds threshold.

    The structural patch motivated by the stitching evasion (§4.2 in the
    report). Reads ``max_window_z`` from the signal table — populated
    upstream by ``audit/sliding_window.py`` against a per-window
    calibration that aggregates over UniProt windows of the same width.
    """
    z = row.get("max_window_z", None)
    if z is None or (isinstance(z, float) and pd.isna(z)):
        # Signal not available (e.g. legacy CSV without sliding-window column).
        return "PASS"
    z = float(z)
    if z >= PPL_BLOCK_Z:
        return "BLOCK"
    if z >= PPL_REVIEW_Z:
        return "REVIEW"
    return "PASS"


PPL_BLOCK_Z_TIGHT = 3.5     # tighter per-window threshold for production-grade FPR
PPL_REVIEW_Z_TIGHT = 2.0


def sliding_window_perplexity_tight(row: pd.Series) -> str:
    """Sliding-window screen with retuned thresholds for production native-FPR.

    The §4.3 default thresholds (z>=3 BLOCK, z>=1.5 REVIEW) inherit the
    whole-sequence calibration's tuning. But per-window NLLs are noisier
    than whole-sequence means (variance scales as 1/W vs 1/L), so the
    same z thresholds over-flag natural sequences with localized high-NLL
    domains (signal peptides, IDRs, viral-like motifs). This variant
    bumps the per-window thresholds by 0.5 sigma to recover
    whole-sequence-like FPR while preserving most of the §4.3 detection
    gain on the stitching attack — see §6.1 limitations bullet 3.
    """
    z = row.get("max_window_z", None)
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "PASS"
    z = float(z)
    if z >= PPL_BLOCK_Z_TIGHT:
        return "BLOCK"
    if z >= PPL_REVIEW_Z_TIGHT:
        return "REVIEW"
    return "PASS"


def or_gate_v3_tight(row: pd.Series) -> str:
    """Production-recommended OR-ensemble: tight-threshold sliding-window + complexity.

    Differences vs ``or_gate_v2``:
      (a) per-window thresholds raised to z>=3.5 BLOCK / z>=2.0 REVIEW
          (compensates for per-window NLL variance vs whole-sequence;
           reduces native-FPR from 10% to 0% on our data); and
      (b) the commec-skipped defense-in-depth fallback is dropped
          (production deployments are expected to have commec configured;
           the fallback is appropriate in development but inflates FPR).
    """
    z_window = row.get("max_window_z", None)
    rep = float(row["repetitiveness"])
    have_window = z_window is not None and not (isinstance(z_window, float) and pd.isna(z_window))
    z_window_f = float(z_window) if have_window else 0.0

    if rep >= LOW_COMPLEXITY_BLOCK:
        return "BLOCK"
    if have_window and z_window_f >= PPL_BLOCK_Z_TIGHT:
        return "BLOCK"
    if rep >= LOW_COMPLEXITY_REVIEW:
        return "REVIEW"
    if have_window and z_window_f >= PPL_REVIEW_Z_TIGHT:
        return "REVIEW"
    return "PASS"


def or_gate_v2(row: pd.Series) -> str:
    """OR-ensemble using sliding-window perplexity instead of whole-sequence.

    Same construction as ``or_gate`` but the perplexity arm is the
    position-resolved gate. Closes the reduction-operator (mosaic
    stitching) failure by construction; closes the one-tailed-test
    (tandem-motif) failure via the complexity arm; remains exposed to
    near-distribution failures (low-temperature ProteinMPNN).
    """
    z_window = row.get("max_window_z", None)
    rep = float(row["repetitiveness"])
    commec_skipped = str(row.get("commec_status", "skipped")) == "skipped"

    have_window = z_window is not None and not (isinstance(z_window, float) and pd.isna(z_window))
    z_window_f = float(z_window) if have_window else 0.0

    if rep >= LOW_COMPLEXITY_BLOCK:
        return "BLOCK"
    if have_window and z_window_f >= PPL_BLOCK_Z:
        return "BLOCK"
    if rep >= LOW_COMPLEXITY_REVIEW:
        return "REVIEW"
    if have_window and z_window_f >= PPL_REVIEW_Z:
        return "REVIEW"
    if commec_skipped and have_window and z_window_f >= PPL_BORDERLINE_Z:
        return "REVIEW"
    return "PASS"


VARIANTS: list[ScreenVariant] = [
    ScreenVariant("perplexity_only", "ESM-2 pseudo-PPL z-gate (z>=3 BLOCK, z>=1.5 REVIEW)", perplexity_only),
    ScreenVariant("complexity_only", "Wootton-Federhen + distinct-k-mer (rep>=0.5 BLOCK, >=0.2 REVIEW)", complexity_only),
    ScreenVariant("or_gate", "PerplexityGuard OR-gate (perplexity OR complexity OR commec)", or_gate),
    ScreenVariant(
        "sliding_window_perplexity",
        "Per-window pLM-PPL z-gate (max-window z>=3 BLOCK, >=1.5 REVIEW)",
        sliding_window_perplexity,
    ),
    ScreenVariant(
        "or_gate_v2",
        "OR-ensemble using sliding-window perplexity (closes reduction-operator failures)",
        or_gate_v2,
    ),
    ScreenVariant(
        "sliding_window_perplexity_tight",
        "Per-window pLM-PPL z-gate with tight thresholds (z>=3.5 BLOCK, z>=2.0 REVIEW)",
        sliding_window_perplexity_tight,
    ),
    ScreenVariant(
        "or_gate_v3_tight",
        "Production-recommended OR-ensemble (sliding-window tight + complexity)",
        or_gate_v3_tight,
    ),
]


def detection_rate(verdicts: pd.Series) -> float:
    """Detection = REVIEW or BLOCK (i.e. anything that is not PASS)."""
    if len(verdicts) == 0:
        return 0.0
    return float((verdicts != "PASS").sum() / len(verdicts))


def block_rate(verdicts: pd.Series) -> float:
    if len(verdicts) == 0:
        return 0.0
    return float((verdicts == "BLOCK").sum() / len(verdicts))


def apply_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Add one verdict column per ScreenVariant onto the CSV.

    Returns a copy with new columns named ``verdict_<variant_name>``.
    """
    out = df.copy()
    for v in VARIANTS:
        out[f"verdict_{v.name}"] = out.apply(v.decide, axis=1)
    return out
