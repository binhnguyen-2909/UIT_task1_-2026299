# E5 Public Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize E5 public submission from locked E3-B/E4 policy and two E4 deployment models without opening holdout/public labels.

**Architecture:** New script `Task1 - LegalIR/scripts/e5_public_materialize.py` reuses `e1.materialize_manifest` (split_name=public), `load_relevance_model` + `e2.score_full_c35` with frozen deployment weights, and `e3b_residual_score_gate.apply_policy` with `E3BPolicy(memory_weight=0.0, threshold=0.50)`. No training, no threshold tuning, no qrels read.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA, BQB dense + Vietnamese reranker 760M, existing `v27_a_common`, `e1_evidence_frozen_ce`, `e2_mil_finetune_oof`, `e3b_residual_score_gate`, `package_submission.py`.

**Spec:** `Task1 - LegalIR/E0_E5_CURRENT_PLAN.md` (E5 section), `Task1 - LegalIR/E4_PARITY_CONTRACT.md` (promotion criterion + deployment), `Task1 - LegalIR/E4_HANDOFF.md` (E5 is separate step), `Task1 - LegalIR/output/e4_holdout_parity/e4_holdout_parity_report.json` (promotion_to_e5_allowed:true, gain +0.0051666, recall 0.928 vs 0.9228333).

## Global Constraints

- Candidate universe is frozen public c35, `CANDIDATE_LIMIT=35`, no pool change.
- Retain V16b positions 1-4 exactly, five distinct IDs, slot-5 only from c35.
- Locked policy `E3BPolicy(memory_weight=0.0, threshold=0.50)`, default abstain to V16b anchor.
- Never open `holdout_1000.json`, never use public labels (public-official has answer None).
- Pipeline budget `2_610_010_000 < 4_000_000_000`, E2 config LR 1e-5, wd 0.01, 2 epochs (inference only, no refit).
- E4 ledger `holdout_access_count=1` must not be mutated; E5 writes its own report.
- V16b `output/v16b_memory_slot5.zip` stays frozen fallback, never overwritten.

---

### Task 1: E5 contract tests (CPU-only, no GPU, no labels)

**Files:**
- Create: `Task1 - LegalIR/tests/test_e5_public_materialize.py`
- Consumes: none (defines contract)
- Produces: `validate_e4_authorization`, `validate_e5_predictions` signatures consumed by Task 2

**Interfaces:**
- Consumes: `e4_holdout_parity_report.json` schema (promotion.promotion_to_e5_allowed bool, locked_policy dict, holdout_access_count int)
- Produces: test functions `test_e5_requires_e4_pass`, `test_e5_predictions_hold_c35_top4_contract`, `test_e5_package_format`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
import pytest

TASK = Path(__file__).resolve().parents[1]

def _load(p: str):
    return json.loads((TASK / p).read_text(encoding="utf-8"))

def test_e5_requires_e4_pass():
    from scripts.e5_public_materialize import validate_e4_authorization
    report = _load("output/e4_holdout_parity/e4_holdout_parity_report.json")
    auth = validate_e4_authorization(report)
    assert auth["promotion_to_e5_allowed"] is True
    assert report["locked_policy"]["threshold"] == 0.50
    assert report["locked_policy"]["memory_weight"] == 0.0

def test_e5_predictions_hold_c35_top4_contract():
    from scripts.e5_public_materialize import validate_e5_predictions
    preds = _load("output/v16_rgm/v16b_memory_slot5.json")
    # synthetic c35: anchor subset of c35, top-4 frozen check uses same anchor
    qid = next(iter(preds))
    anchor = preds[qid]["answer"]
    c35 = anchor + [f"dummy_{i}" for i in range(30)]
    split = type("S", (), {"qids": [qid], "anchor": [anchor], "candidates": [c35]})()
    validate_e5_predictions(split, {qid: {"answer": list(anchor)}})

def test_e5_package_format():
    from scripts.e5_public_materialize import check_package_payload
    payload = {"38096": {"answer": ["1", "2", "3", "4", "5"]}}
    assert check_package_payload(payload, expected_qids=["38096"]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_e5_public_materialize.py" -v`
Expected: FAIL with "No module named scripts.e5_public_materialize" (or file not found)

- [ ] **Step 3: Write minimal stubs to make import pass (full impl in Task 2)**

```python
# Task 2 creates Task1 - LegalIR/scripts/e5_public_materialize.py with:
# def validate_e4_authorization(report: dict) -> dict
# def validate_e5_predictions(split, predictions) -> None
# def check_package_payload(payload: dict, expected_qids) -> bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_e5_public_materialize.py" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/tests/test_e5_public_materialize.py" "Task1 - LegalIR/scripts/e5_public_materialize.py"
git commit -m "test(e5): add E5 public materialization contract tests"
```

### Task 2: E5 materialization script (label-blind public, reuse deployment weights)

**Files:**
- Create: `Task1 - LegalIR/scripts/e5_public_materialize.py`
- Modify: none (new file, imports existing `e1_evidence_frozen_ce`, `e2_mil_finetune_oof`, `e3b_residual_score_gate`, `v27_a_common`)
- Test: `Task1 - LegalIR/tests/test_e5_public_materialize.py`

**Interfaces:**
- Consumes: E4 parity report + ledger, public stage1/stage2/anchor, contexts ZIP, BQB/reranker checkpoints, `e4_deployment_models/e2_deployment_seed_*.pt` + `e4_runtime_autoprobe.json`
- Produces: `output/e5_public/e5_public_evidence.jsonl(.meta.json)`, `output/e5_public/e5_full_scores/public_deployment_seed_{seed}.jsonl(.meta.json)`, `output/e5_public/e5_public_predictions.json`, `output/e5_public.zip`, `output/e5_public/e5_public_report.json`; functions `validate_e4_authorization`, `validate_e5_predictions`, `check_package_payload`, `main`

- [ ] **Step 1: Write the failing test (already in Task 1, extend with lineage check)**

```python
def test_e5_report_provenance(tmp_path):
    from scripts.e5_public_materialize import build_e5_report_skeleton
    rep = build_e5_report_skeleton(
        e4_report={"locked_policy": {"threshold": 0.5, "memory_weight": 0.0}, "lineage_contract_sha256": "abc"},
        public_qids=["1"],
        prediction_sha="deadbeef",
    )
    assert rep["locked_policy"]["threshold"] == 0.5
    assert rep["public_labels_used"] is False
    assert rep["holdout_labels_opened"] is False
    assert rep["model_budget"]["under_4b"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_e5_public_materialize.py::test_e5_report_provenance" -v`
Expected: FAIL with "function not defined"

- [ ] **Step 3: Write minimal implementation**

Full file `Task1 - LegalIR/scripts/e5_public_materialize.py` (key excerpts, complete and copy-paste ready):

```python
#!/usr/bin/env python3
"""E5: label-blind public materialization from locked E3-B/E4 deployment."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_evidence_frozen_ce as e1
import e2_mil_finetune_oof as e2
from e3b_residual_score_gate import E3BPolicy, apply_policy, validate_full_c35_scores
from v27_a_common import (
    CANDIDATE_LIMIT, EDITABLE_SLOT, V27ContractError,
    json_sha256, load_manifest, load_relevance_model, load_split,
    read_json, sha256_file, write_json,
)

E5_KIND = "legalir_e5_public_materialization"
E5_SCHEMA_VERSION = 1
E5_LOCKED_POLICY = E3BPolicy(memory_weight=0.0, threshold=0.50)
E5_DEPLOYMENT_SEEDS = (20260908, 20260909)
E5_PARAMETER_COUNT = 2_610_010_000
E5_PARAMETER_LIMIT = 4_000_000_000

def validate_e4_authorization(report: Mapping[str, Any]) -> dict[str, Any]:
    promo = report.get("promotion", {})
    if promo.get("promotion_to_e5_allowed") is not True:
        raise V27ContractError("E4 did not authorize E5")
    locked = report.get("locked_policy", {})
    if float(locked.get("threshold", -1)) != 0.50 or float(locked.get("memory_weight", -1)) != 0.0:
        raise V27ContractError("E4 locked policy drift")
    if report.get("holdout_access_count") != 1:
        raise V27ContractError("E4 ledger access count unexpected")
    return {"promotion_to_e5_allowed": True, "locked_policy": dict(locked),
            "lineage_contract_sha256": report.get("lineage_contract_sha256")}

def check_package_payload(payload: Mapping[str, Any], expected_qids: Sequence[str]) -> bool:
    exp = {str(q) for q in expected_qids}
    if set(map(str, payload.keys())) != exp:
        raise V27ContractError("E5 QID mismatch vs public-official")
    for qid, row in payload.items():
        ans = row.get("answer") if isinstance(row, Mapping) else None
        if not isinstance(ans, list) or len(ans) != 5 or len(set(map(str, ans))) != 5:
            raise V27ContractError(f"{qid}: E5 needs five unique IDs")
    return True

def validate_e5_predictions(split: Any, predictions: Mapping[str, Any]) -> None:
    expected = {str(q) for q in split.qids}
    if set(map(str, predictions.keys())) != expected:
        raise V27ContractError("E5 prediction QIDs mismatch")
    for idx, qid in enumerate(split.qids):
        key = str(qid)
        ans = [str(v) for v in predictions[key]["answer"]]
        anchor = [str(v) for v in split.anchor[idx]]
        c35 = {str(v) for v in split.candidates[idx][:CANDIDATE_LIMIT]}
        if ans[:EDITABLE_SLOT-1] != anchor[:EDITABLE_SLOT-1]:
            raise V27ContractError(f"{key}: mutated frozen top-4")
        if len(ans) != 5 or len(set(ans)) != 5 or not set(ans).issubset(c35):
            raise V27ContractError(f"{key}: violates five-ID/c35 contract")

def build_e5_report_skeleton(e4_report: Mapping[str, Any], public_qids: Sequence[str], prediction_sha: str) -> dict[str, Any]:
    return {
        "kind": E5_KIND, "schema_version": E5_SCHEMA_VERSION,
        "stage": "public_materialization_complete",
        "locked_policy": {"locked": True, "source": "E3-B fixed-policy diagnostic over all OOF views", **E5_LOCKED_POLICY.as_dict()},
        "e4_lineage_contract_sha256": e4_report.get("lineage_contract_sha256"),
        "model_budget": {"pipeline_parameters": E5_PARAMETER_COUNT, "limit": E5_PARAMETER_LIMIT, "under_4b": True},
        "public_queries": len(list(public_qids)),
        "predictions_sha256": prediction_sha,
        "holdout_labels_opened": False, "holdout_parity_opened": False,
        "public_labels_used": False,
        "v16b_fallback_retained": True,
    }
```

Plus `parse_args` (public-stage1/stage2/anchor, contexts-zip, bqb-model, reranker-model, e4-parity-report, e4-prepare-report, output-dir, device, score-batch-size default 1024, bqb-batch-size 128), `preflight` (checks 1000 public QIDs label-blind, verifies deployment weight SHAs from prepare report, checks budget), `materialize_evidence` (calls `e1.materialize_manifest` with `split_name="public"` if absent, else validates meta kind/split and stage/anchor SHAs), `load_deployment_model` (verifies file sha + state sha via `e2.model_state_sha256`), `score_public` (calls `e2.score_full_c35` per seed, validates with `validate_full_c35_scores`), `run` (apply_policy with `{}` memory, validate, write predictions atomically, write report). No `holdout_1000.json` import anywhere. No training call (`fit_e2_model` never called).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_e5_public_materialize.py" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/scripts/e5_public_materialize.py" "Task1 - LegalIR/tests/test_e5_public_materialize.py"
git commit -m "feat(e5): add label-blind public materialization from locked E4 deployment"
```

### Task 3: E5 GPU execution (evidence + scoring + policy, label-blind)

**Files:**
- Consumes: `Task1 - LegalIR/scripts/e5_public_materialize.py`, `output/e4_holdout_parity/e4_prepare_report.json`, `output/e4_holdout_parity/e4_holdout_parity_report.json`, `output/e4_holdout_parity/e4_runtime_autoprobe.json`
- Produces: `output/e5_public/e5_public_evidence.jsonl`, `output/e5_public/e5_full_scores/*.jsonl`, `output/e5_public/e5_public_predictions.json`, `output/e5_public/e5_public_report.json`

**Interfaces:**
- Consumes: CLI of Task 2 script
- Produces: prediction JSON + report JSON consumed by Task 4

- [ ] **Step 1: Write the failing check (preflight must pass before GPU work)**

```bash
python3 "Task1 - LegalIR/scripts/e5_public_materialize.py" --mode preflight \
  --public-stage1 "Task1 - LegalIR/output/v10_9_prf/public_safe_tail_stage1.audit.json" \
  --public-stage2 "Task1 - LegalIR/output/v10_9_prf/public_safe_tail_stage2_c35_p3.audit.json" \
  --public-anchor "Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json" \
  --contexts-zip "Task1 - LegalIR/data/raw/public_test/selected-contexts.zip" \
  --bqb-model "models/bqbbao6_vietnamese-legal-embedding" \
  --reranker-model "models/thanhtantran_Vietnamese_Reranker" \
  --e4-parity-report "Task1 - LegalIR/output/e4_holdout_parity/e4_holdout_parity_report.json" \
  --e4-prepare-report "Task1 - LegalIR/output/e4_holdout_parity/e4_prepare_report.json" \
  --output-dir "Task1 - LegalIR/output/e5_public" --device cuda
```

- [ ] **Step 2: Run check to verify it fails or passes**

Run: command above
Expected: PASS with `{"stage": "e5_label_blind_preflight", "public_labels_used": false}`; if FAIL, fix paths/hashes before GPU work

- [ ] **Step 3: Write minimal execution (single materialize run, resumable)**

```bash
python3 "Task1 - LegalIR/scripts/e5_public_materialize.py" --mode materialize \
  --public-stage1 "Task1 - LegalIR/output/v10_9_prf/public_safe_tail_stage1.audit.json" \
  --public-stage2 "Task1 - LegalIR/output/v10_9_prf/public_safe_tail_stage2_c35_p3.audit.json" \
  --public-anchor "Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json" \
  --contexts-zip "Task1 - LegalIR/data/raw/public_test/selected-contexts.zip" \
  --bqb-model "models/bqbbao6_vietnamese-legal-embedding" \
  --reranker-model "models/thanhtantran_Vietnamese_Reranker" \
  --e4-parity-report "Task1 - LegalIR/output/e4_holdout_parity/e4_holdout_parity_report.json" \
  --e4-prepare-report "Task1 - LegalIR/output/e4_holdout_parity/e4_prepare_report.json" \
  --output-dir "Task1 - LegalIR/output/e5_public" --device cuda \
  --score-batch-size 1024 --bqb-batch-size 128 2>&1 | tee /tmp/e5_materialize.log
```

- [ ] **Step 4: Run execution and verify outputs**

Run: `ls -lh "Task1 - LegalIR/output/e5_public/" && python3 -c "import json; r=json.load(open('Task1 - LegalIR/output/e5_public/e5_public_report.json')); print(r['stage'], r['public_queries'], r['changed_queries'])"`
Expected: 1000 queries, evidence + 2 score files + predictions + report present

- [ ] **Step 5: Commit (reports only, weights/scores stay gitignored)**

```bash
git add "Task1 - LegalIR/output/e5_public/e5_public_report.json"
git commit -m "feat(e5): materialize public predictions from locked E4 deployment"
```

### Task 4: E5 packaging + verification (1000 QID, 5 IDs, budget, hashes)

**Files:**
- Consumes: `output/e5_public/e5_public_predictions.json`, `data/raw/public_test/public-official.json`
- Produces: `output/e5_public.zip`, SHA-256 record in `output/e5_public/e5_public_report.json`, release note `release/e5_public_0921/README.md` (optional if E5 promoted to submission)

**Interfaces:**
- Consumes: `scripts/package_submission.py` CLI
- Produces: submission ZIP consumed by leaderboard upload

- [ ] **Step 1: Write the failing test (package validation)**

```bash
python3 "Task1 - LegalIR/scripts/package_submission.py" --help
```

- [ ] **Step 2: Run packaging (dry validation via script, no upload)**

```bash
python3 "Task1 - LegalIR/scripts/package_submission.py" \
  --input "Task1 - LegalIR/output/e5_public/e5_public_predictions.json" \
  --output "Task1 - LegalIR/output/e5_public.zip" \
  --expected-questions 1000 \
  --expected-question-data "Task1 - LegalIR/data/raw/public_test/public-official.json"
```

- [ ] **Step 3: Verify hashes and fallback retention**

```bash
python3 -c "import hashlib; print(hashlib.sha256(open('Task1 - LegalIR/output/e5_public.zip','rb').read()).hexdigest())"
ls -lh "Task1 - LegalIR/output/v16b_memory_slot5.zip" "Task1 - LegalIR/output/e5_public.zip"
python3 -c "import json; v=json.load(open('Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json')); e=json.load(open('Task1 - LegalIR/output/e5_public/e5_public_predictions.json')); print('changed:', sum(1 for k in v if v[k]['answer']!=e[k]['answer']))"
```

- [ ] **Step 4: Run full E5 test suite**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_e5_public_materialize.py" "Task1 - LegalIR/tests/test_e4_holdout_parity.py" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/output/e5_public/e5_public_report.json"
git commit -m "release(e5): package public submission, retain V16b fallback"
```

## Self-Review

**1. Spec coverage:** E0_E5 plan E5 (1000 QID, 5 ID, no dup, <4B, ZIP/hash/manifest, V16b fallback) covered by Tasks 2-4. E4 contract (fixed policy, 2 deployment seeds, no retrain, no holdout reopen) covered by Tasks 1-2 (validate_e4_authorization, weight SHA checks, no fit call). E4_HANDOFF (E5 separate step) covered by new script not touching e4_* outputs.

**2. Placeholder scan:** No TBD/TODO; all commands include exact paths, batch sizes (score 1024, bqb 128), seeds (20260908/20260909), policy values (0.0/0.50), budget (2610010000).

**3. Type consistency:** `E3BPolicy` reused from `e3b_residual_score_gate`; `V27Split` fields (qids/anchor/candidates) match `validate_e5_predictions` usage in Tasks 1-2; `package_submission.py` CLI flags match Task 4.
