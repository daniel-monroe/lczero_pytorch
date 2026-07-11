"""Piecewise LR schedule, ported from ../src/lczero_training/training/lr_schedule.py.

A schedule is a list of `LrSegment` rules. The active rule for a given step is the
one with the largest `starting_step <= step`. Within a rule, segments of length
`duration_steps[i]` interpolate `lr[i] -> lr[i+1]` using `transition[i]`
(constant|linear|cosine). A final 0-duration segment holds the last lr forever.
"""
from __future__ import annotations

import math
from typing import List

from ..config import LrSegment

_CONST, _LINEAR, _COSINE = "constant", "linear", "cosine"


def _eval_rule(rule: LrSegment, step: int) -> float:
    lrs = rule.lr
    durations = rule.duration_steps
    transitions = rule.transition
    if not lrs:
        return 0.0
    if len(lrs) == 1:
        return lrs[0]

    local = step - rule.starting_step

    elapsed = 0
    for i, dur in enumerate(durations):
        if i + 1 >= len(lrs):
            break
        a, b = lrs[i], lrs[i + 1]
        trans = transitions[i] if i < len(transitions) else _CONST
        if dur <= 0:
            # open-ended tail: hold a (constant) or this is the final segment.
            if trans == _CONST:
                return a
            return b
        if local < elapsed + dur:
            t = (local - elapsed) / dur
            if trans == _LINEAR:
                return a + (b - a) * t
            if trans == _COSINE:
                return a + 0.5 * (1.0 - math.cos(math.pi * t)) * (b - a)
            return a  # constant
        elapsed += dur
    # past the end: clamp to last lr
    return lrs[-1]


class LrSchedule:
    def __init__(self, segments: List[LrSegment]):
        if not segments:
            segments = [LrSegment(starting_step=0, duration_steps=[0], lr=[0.001])]
        self.segments = sorted(segments, key=lambda s: s.starting_step)

    def __call__(self, step: int) -> float:
        active = self.segments[0]
        for seg in self.segments:
            if seg.starting_step <= step:
                active = seg
            else:
                break
        return _eval_rule(active, step)
