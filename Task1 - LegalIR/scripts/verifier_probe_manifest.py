"""Fixed probe manifest. QIDs hash-sorted, never by stress error."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _task_root() -> Path:
    cwd_base = Path.cwd() / "Task1 - LegalIR"
    if cwd_base.is_dir():
        return cwd_base
    # Fallback: resolve from this file location (scripts/ -> Task1 - LegalIR/)
    return Path(__file__).resolve().parent.parent


def build_probe_manifest(n_probe: int = 60, seed: int = 20260915) -> dict:
    import sys

    base = _task_root()
    sys.path.insert(0, str(base / "scripts"))
    from qwen_c200_common import load_candidate_universe

    uni = load_candidate_universe(
        stage1=base / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=base / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=base / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=base / "experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json",
    )
    ranked = sorted(
        map(str, uni.split.qids),
        key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest(),
    )
    qids = ranked[: int(n_probe)]
    return {
        "seed": int(seed),
        "qids": qids,
        "n_probe": int(n_probe),
        "pairs_max": int(n_probe) * 3 * 2,
    }


def budget_check() -> dict:
    return {
        "complement_pairs_max": 1197 * 3 * 2,
        "public_pairs_max": 1000 * 3 * 2,
        "required_throughput": 7182 / 21600,
    }
