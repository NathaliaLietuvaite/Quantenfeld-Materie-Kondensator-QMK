"""
qmk_v10/q2_30_interlock.py
Bit-exact Q2.30 S^63 Hardware Interlock reference matching V-MAX-12 odos_gate_core_v4.sv (0 LSB).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

ONE_Q2_30: int = 1 << 30
NORM_MIN_Q2_30: int = 1072668082
NORM_MAX_Q2_30: int = 1074815566
BLEED_CEIL_Q2_30: int = 32

TIER_THRESHOLDS_Q2_30 = {
    0: (1020054733, 53687091),  # 0.95 / 0.05
    1: (966367642, 53687091),   # 0.90 / 0.05 (Mosaic)
    2: (1063004406, 10737418),  # 0.99 / 0.01 (Core Lock)
}


@dataclass(frozen=True)
class Q230Verdict:
    tier_mode: int
    norm_sq_q: int
    rcf_raw_q: int
    g3_overlap_q: int
    delta_e_raw_q: int
    bleed_post_q: int
    norm_fault: bool
    mode_fault: bool
    gate_pass: bool
    veto_pin: bool


def _q60_to_q30(acc_q60: int) -> int:
    return (acc_q60 + (1 << 29)) >> 30


def _f64_to_q30(val: float) -> int:
    return int(round(val * ONE_Q2_30))


def evaluate_q230_interlock(
    vec64: Sequence[float],
    anchor64: Sequence[float],
    g3_axis64: Sequence[float],
    tier_mode: int = 0,
) -> Q230Verdict:
    v_q = [_f64_to_q30(x) for x in vec64]
    l_q = [_f64_to_q30(x) for x in anchor64]
    g_q = [_f64_to_q30(x) for x in g3_axis64]

    norm_sq_q = _q60_to_q30(sum(x * x for x in v_q))
    dot_l_q = _q60_to_q30(sum(a * b for a, b in zip(v_q, l_q)))
    dot_g3_q = _q60_to_q30(sum(a * b for a, b in zip(v_q, g_q)))

    rcf_raw_q = _q60_to_q30(dot_l_q * dot_l_q)
    g3_ov_q = abs(dot_g3_q)

    t1 = max(0, ONE_Q2_30 - rcf_raw_q)
    t2 = min(ONE_Q2_30, g3_ov_q)
    delta_e_q = (t1 >> 1) + (t2 >> 1) + (t1 & t2 & 1)

    bleed_acc = 0
    for vi, gi in zip(v_q, g_q):
        stripped = vi - _q60_to_q30(dot_g3_q * gi)
        bleed_acc += stripped * gi
    bleed_post_q = abs(_q60_to_q30(bleed_acc))

    norm_fault = not (NORM_MIN_Q2_30 <= norm_sq_q <= NORM_MAX_Q2_30)
    mode_bits = tier_mode & 0x3
    mode_fault = mode_bits == 3
    rcf_floor, de_ceil = TIER_THRESHOLDS_Q2_30.get(mode_bits, (ONE_Q2_30, 0))

    gate_pass = (
        (not norm_fault)
        and (not mode_fault)
        and (rcf_raw_q >= rcf_floor)
        and (delta_e_q <= de_ceil)
        and (bleed_post_q <= BLEED_CEIL_Q2_30)
    )
    return Q230Verdict(
        tier_mode=mode_bits,
        norm_sq_q=norm_sq_q,
        rcf_raw_q=rcf_raw_q,
        g3_overlap_q=g3_ov_q,
        delta_e_raw_q=delta_e_q,
        bleed_post_q=bleed_post_q,
        norm_fault=norm_fault,
        mode_fault=mode_fault,
        gate_pass=gate_pass,
        veto_pin=not gate_pass,
    )
