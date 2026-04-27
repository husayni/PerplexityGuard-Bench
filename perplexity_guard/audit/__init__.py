"""Audit harness for evaluating pLM-based protein synthesis screens.

This subpackage reframes PerplexityGuard's signals as a benchmark: rather
than presenting one tool, we measure how *each individual signal* (and
their OR-gate) detects a battery of evasion classes. The point is to
identify which attacks each signal misses and which classes are
uncatchable by any current single signal.
"""
