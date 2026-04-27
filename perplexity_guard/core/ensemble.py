"""Ensemble risk scoring: ESM-2 perplexity + commec homology + low-complexity.

Verdict logic (auditable, three-way):

    BLOCK    commec regulated-pathogen homology hit, OR
             perplexity z-score ≥ 3 (≈ 99.9 percentile of natural), OR
             low-complexity fraction ≥ 0.5 (closes the repeat-motif evasion
             where pseudo-perplexity drops below natural).

    REVIEW   perplexity ≥ p95 threshold OR z ≥ 1.5,
             OR commec returned "warn",
             OR low-complexity fraction ≥ 0.2,
             OR commec skipped AND perplexity z ≥ 1.0.

    PASS     otherwise.

Why three signals: the validation phase showed pseudo-perplexity catches
out-of-distribution sequences but is *evaded* by tandem-repeat designs,
which score below natural. A complementary low-complexity check closes that
hole — together they form a two-tailed gate. commec adds an orthogonal
homology axis that catches sequences resembling regulated agents regardless
of how natural they look to a pLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from perplexity_guard.core.complexity import ComplexityResult, wootton_federhen
from perplexity_guard.core.esm_analysis import (
    ESM2Scorer,
    PerplexityResult,
)
from perplexity_guard.core.screening import (
    CommecResult,
    back_translate_aa_to_dna,
    run_commec_screening,
)


VERDICT_PASS = "PASS"
VERDICT_REVIEW = "REVIEW"
VERDICT_BLOCK = "BLOCK"

# Tuned alongside the validation Cohen's d=2.76 / AUROC=0.987 result.
# These apply to ``ComplexityResult.repetitiveness_score`` (0=natural, 1=pure repeat),
# which combines per-window Shannon entropy with distinct-k-mer fraction.
LOW_COMPLEXITY_REVIEW = 0.20   # mild repetition / partial low-complexity region
LOW_COMPLEXITY_BLOCK = 0.50    # tandem repeat or homopolymer


@dataclass
class EnsembleVerdict:
    sequence: str
    sequence_length: int
    perplexity: PerplexityResult
    commec: CommecResult
    complexity: ComplexityResult
    verdict: str
    flags: list[str]
    explanation: str
    runtime_seconds: float
    dna_sequence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "verdict": self.verdict,
            "flags": self.flags,
            "explanation": self.explanation,
            "runtime_seconds": self.runtime_seconds,
            "perplexity": {
                "value": self.perplexity.pseudo_perplexity,
                "zscore": self.perplexity.zscore,
                "p95_threshold": self.perplexity.p95_threshold,
                "is_high": self.perplexity.is_high,
                "runtime_seconds": self.perplexity.runtime_seconds,
            },
            "complexity": {
                "mean": self.complexity.mean_complexity,
                "min": self.complexity.min_complexity,
                "low_complexity_fraction": self.complexity.low_complexity_fraction,
                "distinct_3mer_fraction": self.complexity.distinct_3mer_fraction,
                "distinct_5mer_fraction": self.complexity.distinct_5mer_fraction,
                "repetitiveness_score": self.complexity.repetitiveness_score,
                "is_repetitive": self.complexity.is_repetitive,
                "window_size": self.complexity.window_size,
            },
            "commec": self.commec.to_dict(),
            "metadata": self.metadata,
        }


def ensemble_screen(
    aa_sequence: str,
    scorer: ESM2Scorer,
    *,
    run_commec: bool = True,
    name: str = "query",
    metadata: dict[str, Any] | None = None,
    work_dir: Path | None = None,
    complexity_window: int = 12,
) -> EnsembleVerdict:
    """End-to-end screen of one amino acid sequence.

    Parameters
    ----------
    aa_sequence : str
        Protein sequence (single-letter codes; whitespace stripped).
    scorer : ESM2Scorer
        Pre-loaded ESM-2 scorer — reuse across calls to avoid model reloads.
    run_commec : bool
        If False, skip homology screening (perplexity + complexity only).
    complexity_window : int
        Sliding-window size for the Wootton-Federhen low-complexity check.
    """
    t0 = time.time()
    seq = aa_sequence.upper().strip().replace(" ", "")
    if not seq:
        raise ValueError("empty sequence")

    perp = scorer.score_sequence(seq)
    complexity = wootton_federhen(seq, window=complexity_window)
    dna = back_translate_aa_to_dna(seq)
    if run_commec:
        commec = run_commec_screening(dna, name=name, work_dir=work_dir)
    else:
        commec = CommecResult(
            status="skipped",
            verdict="homology screening disabled by caller",
            runtime_seconds=0.0,
            skipped_reason="run_commec=False",
        )

    verdict, flags, explanation = _decide(perp, commec, complexity)
    return EnsembleVerdict(
        sequence=seq,
        sequence_length=len(seq),
        perplexity=perp,
        commec=commec,
        complexity=complexity,
        verdict=verdict,
        flags=flags,
        explanation=explanation,
        runtime_seconds=time.time() - t0,
        dna_sequence=dna,
        metadata=metadata or {},
    )


def _decide(
    perp: PerplexityResult,
    commec: CommecResult,
    complexity: ComplexityResult,
) -> tuple[str, list[str], str]:
    flags: list[str] = []
    reasons: list[str] = []

    if commec.is_hit:
        if commec.status == "flag":
            flags.append("homology_regulated")
        else:
            flags.append("homology_warn")
        reasons.append(f"commec: {commec.verdict}")

    if perp.zscore >= 3.0:
        flags.append("perplexity_extreme")
        reasons.append(
            f"PPL={perp.pseudo_perplexity:.2f} (z={perp.zscore:+.2f}, ≥3σ from natural)"
        )
    elif perp.is_high or perp.zscore >= 1.5:
        flags.append("perplexity_high")
        reasons.append(
            f"PPL={perp.pseudo_perplexity:.2f} (z={perp.zscore:+.2f}, "
            f"p95={perp.p95_threshold:.2f})"
        )
    else:
        reasons.append(
            f"PPL={perp.pseudo_perplexity:.2f} (z={perp.zscore:+.2f}, within natural range)"
        )

    rep_score = complexity.repetitiveness_score
    lc_frac = complexity.low_complexity_fraction
    d3 = complexity.distinct_3mer_fraction
    if rep_score >= LOW_COMPLEXITY_BLOCK:
        flags.append("low_complexity_extreme")
        reasons.append(
            f"repetitiveness={rep_score:.2f} (entropy-lc-frac={lc_frac:.2f}, "
            f"distinct-3mer={d3:.2f}); tandem-repeat / homopolymer signature"
        )
    elif rep_score >= LOW_COMPLEXITY_REVIEW:
        flags.append("low_complexity_high")
        reasons.append(
            f"repetitiveness={rep_score:.2f} "
            f"(distinct-3mer={d3:.2f}, mean entropy {complexity.mean_complexity:.2f})"
        )
    else:
        reasons.append(
            f"complexity ok (mean entropy {complexity.mean_complexity:.2f}, "
            f"distinct-3mer={d3:.2f}, repetitiveness={rep_score:.2f})"
        )

    if commec.is_skipped:
        flags.append("homology_skipped")
        reasons.append(f"commec skipped: {commec.skipped_reason}")
    elif commec.is_clear:
        reasons.append("commec: no regulated-pathogen homology")

    block_signals = {"homology_regulated", "perplexity_extreme", "low_complexity_extreme"}
    review_signals = {"homology_warn", "perplexity_high", "low_complexity_high"}

    if any(f in flags for f in block_signals):
        verdict = VERDICT_BLOCK
    elif any(f in flags for f in review_signals):
        verdict = VERDICT_REVIEW
    elif "homology_skipped" in flags and perp.zscore >= 1.0:
        verdict = VERDICT_REVIEW
        flags.append("perplexity_borderline_no_homology")
    else:
        verdict = VERDICT_PASS

    explanation = "; ".join(reasons)
    return verdict, flags, explanation


_VERDICT_COLORS = {
    VERDICT_PASS: "#1a9641",      # green
    VERDICT_REVIEW: "#fdae61",    # amber
    VERDICT_BLOCK: "#d7191c",     # red
}


def risk_color(verdict: str) -> str:
    return _VERDICT_COLORS.get(verdict, "#666666")
