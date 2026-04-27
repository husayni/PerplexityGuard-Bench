"""Unit tests for PerplexityGuard core logic.

Run with:
    uv run python -m unittest perplexity_guard.tests.test_validation
"""

from __future__ import annotations

import unittest

from perplexity_guard.core.complexity import wootton_federhen
from perplexity_guard.core.ensemble import (
    VERDICT_BLOCK,
    VERDICT_PASS,
    VERDICT_REVIEW,
    _decide,
)
from perplexity_guard.core.esm_analysis import Calibration, PerplexityResult
from perplexity_guard.core.screening import (
    CommecResult,
    back_translate_aa_to_dna,
)


def _ppl(value: float, calib: Calibration) -> PerplexityResult:
    return PerplexityResult(
        sequence_length=100,
        pseudo_perplexity=value,
        mean_nll=0.0,  # unused in decisions
        zscore=calib.zscore(value),
        p95_threshold=calib.p95,
        p99_threshold=calib.p99,
        is_high=value >= calib.p95,
        runtime_seconds=0.0,
    )


def _natural_complexity():
    """Plausible complexity result for a natural sequence."""
    return wootton_federhen("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")


def _repeat_complexity():
    """Tandem-repeat sequence — should trip the low-complexity detector."""
    return wootton_federhen("GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA")


class TestBackTranslation(unittest.TestCase):
    def test_known_residues_round_trip(self):
        dna = back_translate_aa_to_dna("MAGI", append_stop=False)
        # M=ATG, A=GCC, G=GGC, I=ATC
        self.assertEqual(dna, "ATGGCCGGCATC")

    def test_appends_stop_codon(self):
        dna = back_translate_aa_to_dna("M", append_stop=True)
        self.assertTrue(dna.endswith("TAA"))
        self.assertEqual(len(dna), 6)

    def test_skips_terminator_in_input(self):
        dna_with_star = back_translate_aa_to_dna("M*", append_stop=False)
        dna_without = back_translate_aa_to_dna("M", append_stop=False)
        self.assertEqual(dna_with_star, dna_without)

    def test_rejects_non_standard_residue(self):
        with self.assertRaises(ValueError):
            back_translate_aa_to_dna("MAGB")  # B = aspartate or asparagine, ambiguous

    def test_lowercase_is_normalised(self):
        a = back_translate_aa_to_dna("magi", append_stop=False)
        b = back_translate_aa_to_dna("MAGI", append_stop=False)
        self.assertEqual(a, b)


class TestEnsembleDecision(unittest.TestCase):
    def setUp(self):
        # Calibration that mirrors the validation: natural mean=11.54 std=3.65 p95=17.0.
        self.calib = Calibration(
            model="test", mean=11.54, std=3.65, p95=17.0, p99=18.5, n=60, source="test"
        )
        self.commec_skipped = CommecResult(
            status="skipped", verdict="not configured",
            runtime_seconds=0.0, skipped_reason="test",
        )
        self.commec_clear = CommecResult(
            status="clear", verdict="no homology", runtime_seconds=0.0,
        )
        self.commec_warn = CommecResult(
            status="warn", verdict="review recommended", runtime_seconds=0.0,
        )
        self.commec_flag = CommecResult(
            status="flag", verdict="regulated pathogen", runtime_seconds=0.0,
            hits=[{"db": "regulated", "id": "fake"}],
        )
        self.cx_natural = _natural_complexity()
        self.cx_repeat = _repeat_complexity()

    def test_natural_passes_when_commec_clear(self):
        v, _, _ = _decide(_ppl(11.0, self.calib), self.commec_clear, self.cx_natural)
        self.assertEqual(v, VERDICT_PASS)

    def test_natural_passes_when_commec_skipped(self):
        # PPL near the mean → z ≈ 0; skipped commec must not flip verdict.
        v, flags, _ = _decide(_ppl(11.0, self.calib), self.commec_skipped, self.cx_natural)
        self.assertEqual(v, VERDICT_PASS)
        self.assertIn("homology_skipped", flags)

    def test_high_perplexity_triggers_review(self):
        v, flags, _ = _decide(_ppl(18.0, self.calib), self.commec_clear, self.cx_natural)
        self.assertEqual(v, VERDICT_REVIEW)
        self.assertIn("perplexity_high", flags)

    def test_extreme_perplexity_triggers_block(self):
        # z = (25-11.54)/3.65 ≈ 3.69
        v, flags, _ = _decide(_ppl(25.0, self.calib), self.commec_clear, self.cx_natural)
        self.assertEqual(v, VERDICT_BLOCK)
        self.assertIn("perplexity_extreme", flags)

    def test_regulated_homology_blocks_even_natural_perplexity(self):
        v, flags, _ = _decide(_ppl(11.0, self.calib), self.commec_flag, self.cx_natural)
        self.assertEqual(v, VERDICT_BLOCK)
        self.assertIn("homology_regulated", flags)

    def test_warn_homology_triggers_review(self):
        v, flags, _ = _decide(_ppl(11.0, self.calib), self.commec_warn, self.cx_natural)
        self.assertEqual(v, VERDICT_REVIEW)
        self.assertIn("homology_warn", flags)

    def test_borderline_with_skipped_commec_triggers_review(self):
        # PPL=15 → z ≈ 0.95 (below 1.5 high threshold) but ≥ 1.0 borderline.
        # Without homology to confirm, the safer default is REVIEW.
        v, flags, _ = _decide(_ppl(15.5, self.calib), self.commec_skipped, self.cx_natural)
        self.assertEqual(v, VERDICT_REVIEW)
        self.assertIn("perplexity_borderline_no_homology", flags)

    def test_repeat_motif_blocks_despite_low_perplexity(self):
        # The whole point of the complexity gate: PPL=1.0 (low, looks "natural"),
        # but the sequence is a tandem repeat → block on complexity.
        v, flags, _ = _decide(
            _ppl(1.0, self.calib), self.commec_clear, self.cx_repeat,
        )
        self.assertEqual(v, VERDICT_BLOCK)
        self.assertIn("low_complexity_extreme", flags)


class TestComplexity(unittest.TestCase):
    def test_natural_sequence_has_high_complexity(self):
        r = _natural_complexity()
        self.assertGreater(r.mean_complexity, 0.7)
        self.assertLess(r.low_complexity_fraction, 0.1)

    def test_tandem_repeat_has_low_complexity(self):
        r = _repeat_complexity()
        self.assertLess(r.mean_complexity, 0.4)
        self.assertGreaterEqual(r.low_complexity_fraction, 0.5)

    def test_homopolymer_has_zero_complexity(self):
        r = wootton_federhen("AAAAAAAAAAAAAAAAAAAA")
        self.assertEqual(r.min_complexity, 0.0)
        self.assertEqual(r.low_complexity_fraction, 1.0)

    def test_empty_sequence_returns_default(self):
        r = wootton_federhen("")
        self.assertEqual(r.sequence_length, 0)
        self.assertEqual(r.n_windows, 0)


if __name__ == "__main__":
    unittest.main()
