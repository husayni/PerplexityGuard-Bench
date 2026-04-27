"""Wootton-Federhen low-complexity detector.

Closes the repeat-motif evasion path identified in the validation phase:
sequences like ``GAGAGAGA…`` score *below* natural pseudo-perplexity (the
model predicts the next residue trivially), so a one-tailed perplexity gate
lets them through. A complementary entropy check flags such designs.

The Wootton & Federhen (1993) approach computes per-window compositional
complexity:

    K(window) = (1/L) * log_N (L! / Π n_i!)         (their formulation)

equivalently, the Shannon entropy of the residue distribution normalized by
the maximum possible entropy. This implementation uses the entropy form
because it is numerically stable and matches the modern SEG / DUST
intuitions.

Per-sequence aggregates returned:
    - ``mean_complexity`` : average normalised entropy across windows in [0, 1]
    - ``low_complexity_fraction`` : fraction of windows with normalised
      entropy below ``low_complexity_threshold`` (default 0.5)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


_MAX_ENTROPY_BITS = math.log2(20)  # 20 standard amino acids


@dataclass
class ComplexityResult:
    sequence_length: int
    window_size: int
    mean_complexity: float            # average normalized entropy across windows ∈ [0, 1]
    min_complexity: float             # lowest single-window complexity
    low_complexity_fraction: float    # fraction of windows below entropy threshold
    threshold: float
    n_windows: int
    # Distinct-k-mer fraction = unique k-mers / total k-mers; small ≈ tandem repeat.
    distinct_3mer_fraction: float = 1.0
    distinct_5mer_fraction: float = 1.0
    # is_repetitive captures the OR over our two complexity signals:
    # * low Shannon entropy in a window (catches homopolymers, low-residue runs)
    # * very few distinct k-mers (catches tandem repeats with diverse motifs)
    is_repetitive: bool = False
    repetitiveness_score: float = 0.0  # 0=natural, 1=pure repeat


def _window_normalized_entropy(window: str) -> float:
    """Shannon entropy of residue distribution, normalised by log2(min(L, 20))."""
    n = len(window)
    if n <= 1:
        return 1.0
    counts = Counter(window)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    # Normalise: max attainable entropy is log2(min(window_size, 20)).
    h_max = math.log2(min(n, 20))
    if h_max <= 0:
        return 1.0
    return h / h_max


def _distinct_kmer_fraction(seq: str, k: int) -> float:
    """Unique k-mers / total k-mers. ~1.0 for natural, ≪1.0 for tandem repeats."""
    if len(seq) < k:
        return 1.0
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    if not kmers:
        return 1.0
    return len(set(kmers)) / len(kmers)


def wootton_federhen(
    sequence: str,
    window: int = 12,
    low_complexity_threshold: float = 0.5,
    repeat_kmer_threshold: float = 0.5,
) -> ComplexityResult:
    """Compute per-sequence low-complexity statistics.

    Two complementary signals:

    1. **Per-window normalised Shannon entropy** (Wootton-Federhen 1993).
       Catches homopolymers and runs of 1–3 distinct residues. Misses
       tandem repeats where the motif has 4+ distinct residues because the
       per-window entropy stays high.

    2. **Distinct-k-mer fraction**. For ``k=3`` and ``k=5``, computes
       ``|{kmers}| / total_kmers``. Pure tandem-repeat tiles drop to
       ``≈k/L`` (single-digit percent) regardless of motif diversity.

    A sequence is flagged as ``is_repetitive`` if EITHER signal trips —
    that's the OR-gate that closes the repeat-motif evasion identified in
    the validation phase, where pure tiles scored *below* natural
    pseudo-perplexity.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (uppercase 20-AA alphabet recommended).
    window : int
        Sliding window for the entropy signal. 12 matches the SEG default.
    low_complexity_threshold : float
        Windows with normalised entropy below this count as low-complexity.
    repeat_kmer_threshold : float
        ``distinct_3mer_fraction`` below this (default 0.5) triggers the
        tandem-repeat flag. Natural sequences sit ~0.95+; pure tiles ~0.1.
    """
    seq = sequence.upper()
    L = len(seq)
    if L == 0:
        return ComplexityResult(
            sequence_length=0, window_size=window, mean_complexity=1.0,
            min_complexity=1.0, low_complexity_fraction=0.0,
            threshold=low_complexity_threshold, n_windows=0,
            distinct_3mer_fraction=1.0, distinct_5mer_fraction=1.0,
            is_repetitive=False, repetitiveness_score=0.0,
        )
    w = min(window, L)
    n_windows = max(L - w + 1, 1)
    complexities = [
        _window_normalized_entropy(seq[i : i + w]) for i in range(n_windows)
    ]
    mean_c = sum(complexities) / n_windows
    min_c = min(complexities)
    low_count = sum(1 for c in complexities if c < low_complexity_threshold)
    lc_frac = float(low_count / n_windows)

    d3 = _distinct_kmer_fraction(seq, 3)
    d5 = _distinct_kmer_fraction(seq, 5)
    # Repetitiveness score: take the worst (lowest) k-mer signal, plus the
    # entropy signal. 1 = pure repeat, 0 = natural.
    kmer_repeat_score = max(0.0, 1.0 - min(d3, d5) / repeat_kmer_threshold)
    entropy_repeat_score = lc_frac
    repetitiveness = max(kmer_repeat_score, entropy_repeat_score)
    is_repetitive = (
        lc_frac >= 0.20
        or d3 < repeat_kmer_threshold
        or d5 < repeat_kmer_threshold
    )

    return ComplexityResult(
        sequence_length=L,
        window_size=w,
        mean_complexity=float(mean_c),
        min_complexity=float(min_c),
        low_complexity_fraction=lc_frac,
        threshold=low_complexity_threshold,
        n_windows=int(n_windows),
        distinct_3mer_fraction=float(d3),
        distinct_5mer_fraction=float(d5),
        is_repetitive=bool(is_repetitive),
        repetitiveness_score=float(min(repetitiveness, 1.0)),
    )
