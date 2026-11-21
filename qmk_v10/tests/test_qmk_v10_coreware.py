"""
qmk_v10/tests/test_qmk_v10_coreware.py
Executable verification suite for QMK-RVC-V10 Coreware Bridge (REGISTER_M).
"""
from __future__ import annotations

import ast
import math
import pathlib
import unittest

from qmk_v10.q2_30_interlock import evaluate_q230_interlock
from qmk_v10.voxel_l2_manifold import (
    QMKVoxelManifold,
    VOXEL_COUNT,
    VoxelMirrorNode,
)


class TestQmkV10Coreware(unittest.TestCase):
    def _basis_vectors(self):
        l_vec = [1.0 / math.sqrt(64.0)] * 64
        g3_vec = [1.0 / math.sqrt(64.0) if i % 2 == 0 else -1.0 / math.sqrt(64.0) for i in range(64)]
        return l_vec, g3_vec

    def test_01_q230_interlock_canonical_pass_and_g3_veto(self) -> None:
        l_vec, g3_vec = self._basis_vectors()
        v_pass = evaluate_q230_interlock(l_vec, l_vec, g3_vec, tier_mode=0)
        self.assertTrue(v_pass.gate_pass)
        self.assertFalse(v_pass.veto_pin)
        v_veto = evaluate_q230_interlock(g3_vec, l_vec, g3_vec, tier_mode=0)
        self.assertFalse(v_veto.gate_pass)
        self.assertTrue(v_veto.veto_pin)

    def test_02_antipodal_recoil_cancellation_1024_voxels(self) -> None:
        m = QMKVoxelManifold()
        self.assertEqual(len(m.k_vectors), VOXEL_COUNT)
        self.assertLess(m.recoil_norm, 1e-12)

    def test_03_p0_single_writer_enforcement(self) -> None:
        m = QMKVoxelManifold()
        ok_p2, msg = m.commit_p0("P2", "phase_shift", 0.1)
        self.assertFalse(ok_p2)
        self.assertEqual(msg, "P0_WRITE_PRIVILEGE_REQUIRED")
        ok_p0, sha = m.commit_p0("P0", "phase_shift", 0.0)
        self.assertTrue(ok_p0)
        self.assertEqual(len(sha), 64)

    def test_04_stale_epoch_veto_and_autonomous_resync(self) -> None:
        m = QMKVoxelManifold()
        node = VoxelMirrorNode("QMK-VOXEL-NODE-07")
        node.sync_from(m.epoch, m.state_sha256)
        self.assertTrue(node.check_coherence(m.epoch, m.state_sha256).coherent)

        m.commit_p0("P0", "gstar_epoch_update", 1)
        stale = node.check_coherence(m.epoch, m.state_sha256)
        self.assertFalse(stale.coherent)
        self.assertEqual(stale.status, "STALE_EPOCH_VETO")

        node.sync_from(m.epoch, m.state_sha256)
        self.assertTrue(node.check_coherence(m.epoch, m.state_sha256).coherent)

    def test_05_hybrid_pq_or_ecdsa_provenance_verified(self) -> None:
        m = QMKVoxelManifold()
        self.assertIn(m.signer.signature_mode, ("HYBRID_ECDSA_P256_ML-DSA-65", "CLASSICAL_ECDSA_P256_ONLY"))
        self.assertTrue(m.signer.sign_and_verify(b"QMK-V10-EPOCH"))

    def test_06_tier3_glitch_trap_fail_closed(self) -> None:
        l_vec, g3_vec = self._basis_vectors()
        v_t3 = evaluate_q230_interlock(l_vec, l_vec, g3_vec, tier_mode=3)
        self.assertTrue(v_t3.mode_fault)
        self.assertFalse(v_t3.gate_pass)
        self.assertTrue(v_t3.veto_pin)

    def test_07_ast_hygiene(self) -> None:
        root = pathlib.Path(__file__).resolve().parent.parent
        for fn in ("q2_30_interlock.py", "voxel_l2_manifold.py"):
            src = (root / fn).read_text(encoding="utf-8")
            self.assertNotIn("\t", src)
            self.assertNotIn(";", src)
            ast.parse(src)


if __name__ == "__main__":
    unittest.main()
