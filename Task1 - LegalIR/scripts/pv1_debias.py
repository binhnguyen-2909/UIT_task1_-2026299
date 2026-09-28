#!/usr/bin/env python3
"""PV1 position-debias helper, eval-only.

Frozen PV1 cache shows extreme Document1 bias: argmax A ~72-74% in both
orientations, only 7/3591 challenger pairs pass the strict both-orders
argmax gate. This module scores challenger preference averaged across
positions, cancelling additive position bias without touching GPU cache.

For incumbent_first: Doc1=incumbent (A), Doc2=challenger (B).
For challenger_first: Doc1=challenger (A), Doc2=incumbent (B).
Debiased score = 0.5 * [(fwd_B - fwd_A) + (rev_A - rev_B)].
Positive means challenger preferred on average. Pure Doc1 bias cancels
to exactly zero. D/C-dominant rows score near zero and abstain.
Complement-only; never opens holdout or public labels.
"""

from __future__ import annotations

from typing import Mapping


def debiased_challenger_score(
    forward: Mapping[str, float],
    reverse: Mapping[str, float],
) -> float:
    """Average challenger-minus-incumbent margin across both orders."""
    fwd_a = float(forward["A"])
    fwd_b = float(forward["B"])
    rev_a = float(reverse["A"])
    rev_b = float(reverse["B"])
    return float(0.5 * ((fwd_b - fwd_a) + (rev_a - rev_b)))
