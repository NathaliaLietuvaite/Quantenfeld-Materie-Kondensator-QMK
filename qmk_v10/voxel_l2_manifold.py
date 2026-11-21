"""
qmk_v10/voxel_l2_manifold.py
QMK-RVC-V10 1,024-Voxel Antipodal Array & Epoch-Versioned 2.00 MiB L2-SRAM Manifold.
Supports live ECDSA P-256 + optional FIPS 204 ML-DSA-65 (liboqs) hybrid signing.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

try:
    import oqs as _oqs_lib
except ImportError:
    _oqs_lib = None

MAX_L2_ENCLAVE_BYTES: int = 2 * 1024 * 1024
VOXEL_COUNT: int = 1024


@dataclass(frozen=True)
class VoxelReadStatus:
    coherent: bool
    status: str
    local_epoch: int
    authoritative_epoch: int


def build_antipodal_voxel_k_vectors(n_voxels: int = VOXEL_COUNT) -> List[Tuple[float, float, float]]:
    """Construct n_voxels paired antipodally (+k, -k) so net recoil sum(k) == (0, 0, 0)."""
    half = n_voxels // 2
    vecs: List[Tuple[float, float, float]] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(half):
        y = 1.0 - (i / max(1, half - 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        kx, ky, kz = math.cos(theta) * r, y, math.sin(theta) * r
        vecs.append((kx, ky, kz))
        vecs.append((-kx, -ky, -kz))
    return vecs


def compute_net_recoil_norm(k_vectors: List[Tuple[float, float, float]]) -> float:
    sx = sum(v[0] for v in k_vectors)
    sy = sum(v[1] for v in k_vectors)
    sz = sum(v[2] for v in k_vectors)
    return math.sqrt(sx * sx + sy * sy + sz * sz)


class QMKHybridSigner:
    """ECDSA P-256 signer with automatic FIPS 204 ML-DSA-65 hybrid upgrade when liboqs is present."""

    def __init__(self) -> None:
        self._sk = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self._pk = self._sk.public_key()
        self.pq_active: bool = False
        self._oqs: Any = None
        self._pq_pk: Optional[bytes] = None
        if _oqs_lib is not None:
            try:
                sig = _oqs_lib.Signature("ML-DSA-65")
                self._pq_pk = sig.generate_keypair()
                self._oqs = sig
                self.pq_active = True
            except Exception:
                self.pq_active = False

    @property
    def signature_mode(self) -> str:
        return "HYBRID_ECDSA_P256_ML-DSA-65" if self.pq_active else "CLASSICAL_ECDSA_P256_ONLY"

    def sign_and_verify(self, payload: bytes) -> bool:
        digest = hashlib.sha256(payload).digest()
        ec_sig = self._sk.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        self._pk.verify(ec_sig, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        if not self.pq_active or self._oqs is None or self._pq_pk is None:
            return True
        pq_sig = self._oqs.sign(payload)
        verifier = _oqs_lib.Signature("ML-DSA-65")
        return bool(verifier.verify(payload, pq_sig, self._pq_pk))


class VoxelMirrorNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.local_epoch: int = 0
        self.local_sha256: str = ""

    def sync_from(self, epoch: int, state_sha256: str) -> None:
        self.local_epoch = epoch
        self.local_sha256 = state_sha256

    def check_coherence(self, auth_epoch: int, auth_sha256: str) -> VoxelReadStatus:
        if self.local_epoch != auth_epoch or self.local_sha256 != auth_sha256:
            return VoxelReadStatus(False, "STALE_EPOCH_VETO", self.local_epoch, auth_epoch)
        return VoxelReadStatus(True, "LOCAL_L2_COHERENT_PASS", self.local_epoch, auth_epoch)


class QMKVoxelManifold:
    """Authoritative P0-gated 2.00 MiB L2-SRAM manifold for 1,024 QMK-V10 voxels."""

    def __init__(self, genesis_anchor: str = "bb17ede97aa45c76ef9afa63cb457f7d684e5ede40b0432b2ecf3b6b9d3e03a3") -> None:
        self.epoch: int = 1
        self.k_vectors = build_antipodal_voxel_k_vectors(VOXEL_COUNT)
        self.recoil_norm = compute_net_recoil_norm(self.k_vectors)
        self._state: Dict[str, Any] = {
            "genesis_anchor": genesis_anchor,
            "voxel_count": len(self.k_vectors),
            "recoil_norm": self.recoil_norm,
        }
        self.signer = QMKHybridSigner()
        self.state_sha256 = self._recompute_sha()

    def _recompute_sha(self) -> str:
        raw = json.dumps({"epoch": self.epoch, "state": self._state}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def commit_p0(self, caller_tier: str, key: str, value: Any) -> Tuple[bool, str]:
        if caller_tier != "P0":
            return False, "P0_WRITE_PRIVILEGE_REQUIRED"
        cand = dict(self._state)
        cand[key] = value
        raw = json.dumps({"epoch": self.epoch + 1, "state": cand}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_L2_ENCLAVE_BYTES:
            return False, "L2_ENCLAVE_2MIB_EXCEEDED"
        self._state = cand
        self.epoch += 1
        self.state_sha256 = hashlib.sha256(raw).hexdigest()
        assert self.signer.sign_and_verify(raw)
        return True, self.state_sha256
