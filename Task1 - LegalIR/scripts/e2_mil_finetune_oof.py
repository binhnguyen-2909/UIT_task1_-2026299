"""E2 complement-only OOF fine-tuning on the sealed E1 evidence manifest.

E1 established that the new evidence pack is useful for frozen CE reading but
cannot replace the V16b decision path wholesale.  E2 therefore trains only on
document qrels within the frozen c35 universe.  Passage identity remains
latent: document scores are soft MIL aggregates over the three/four passages.

The script never accepts holdout or public paths.  Every outer fold trains a
fresh student and scores only its unseen validation queries.  An inner split is
used for early stopping; no outer labels select epoch, calibration or policy.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    PairEncodingCache,
    V27ContractError,
    _raw_model_score,
    build_regime_groups,
    group_folds,
    load_manifest,
    load_relevance_model,
    load_split,
    normalized_text,
    row_blocks,
    sha256_file,
    temperature_logsumexp,
    transition_metrics,
    write_json,
)


E2_KIND = "legalir_e2_mil_relevance_boundary_oof"
E2_SCHEMA_VERSION = 1
E2_MAX_LENGTH = 768
E2_TEMPERATURE = 0.5
E2_LOSS_WEIGHTS = {
    "relevance": 0.55,
    "boundary": 0.30,
    "anchor": 0.15,
}
E2_DEFAULT_SEEDS = (20260908, 20260909)
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")
CHECKPOINT_KIND = "legalir_e2_mil_oof_fold_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class E2Data:
    """The immutable complement inputs shared by E2 and E3-B refits."""

    meta: Mapping[str, Any]
    split: Any
    manifest: Mapping[tuple[str, str], Mapping[str, Any]]
    frozen_scores: Mapping[tuple[str, str], float | None]
    examples: Mapping[str, Sequence[Mapping[str, Any]]]
    groups_by_regime: Mapping[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--e1-manifest-meta", type=Path, required=True)
    parser.add_argument("--e1-frozen-scores", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, help="final OOF report; defaults inside output-dir")
    parser.add_argument("--regimes", nargs="+", default=["standard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(E2_DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="recompute activations to fit smaller VRAM GPUs",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="use available VRAM for faster backward passes",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--memory-probe", action="store_true", help="run one exact optimizer step to verify CUDA memory before OOF training")
    parser.add_argument("--checkpoint-path", type=Path, help="atomic fold-level progress file; defaults inside output-dir")
    parser.add_argument("--resume", dest="resume", action="store_true", help="reuse completed folds from a matching checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="discard any compatible fold checkpoint and start fresh")
    parser.set_defaults(resume=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_complement_path(path: Path, label: str) -> None:
    value = str(path).replace("\\", "/").casefold()
    if any(term in value for term in FORBIDDEN_TERMS):
        raise ValueError(f"E2 is complement-only; {label} may not reference holdout/public: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_path or args.output_dir / "e2_mil_oof_progress.json"


def _report_path(args: argparse.Namespace) -> Path:
    return args.report_path or args.output_dir / "e2_mil_oof_report.json"


def _checkpoint_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint every choice that can change an OOF fold result.

    We checkpoint only at the fold boundary.  Reusing a fold after any change
    in data, model provenance, split regime or optimization configuration is
    prohibited rather than silently mixing incomparable predictions.
    """

    return {
        "split": "complement",
        "stage1_sha256": sha256_file(args.stage1),
        "stage2_sha256": sha256_file(args.stage2),
        "anchor_sha256": sha256_file(args.anchor),
        "reference_sha256": sha256_file(args.reference),
        "e1_manifest_sha256": sha256_file(args.e1_manifest),
        "e1_manifest_meta_sha256": sha256_file(args.e1_manifest_meta),
        "e1_frozen_scores_sha256": sha256_file(args.e1_frozen_scores),
        "model_config_sha256": sha256_file(args.reranker_model / "config.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "regimes": list(args.regimes),
        "seeds": list(args.seeds),
        "epochs": int(args.epochs),
        "micro_batch": int(args.micro_batch),
        "gradient_accumulation": int(args.gradient_accumulation),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "score_batch_size": int(args.score_batch_size),
        "max_negatives": int(args.max_negatives),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "max_length": E2_MAX_LENGTH,
        "temperature": E2_TEMPERATURE,
        "loss_weights": E2_LOSS_WEIGHTS,
    }


def _load_or_create_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(args)
    contract = _checkpoint_contract(args)
    contract_sha256 = _stable_hash(contract)
    if args.resume and path.exists():
        state = _read_json(path)
        if state.get("kind") != CHECKPOINT_KIND or state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise V27ContractError(f"Invalid E2 checkpoint: {path}")
        if state.get("contract_sha256") != contract_sha256:
            raise V27ContractError(
                "E2 checkpoint contract differs from this run; use --no-resume to start fresh"
            )
        if not isinstance(state.get("completed_folds"), dict):
            raise V27ContractError(f"E2 checkpoint has no completed_folds mapping: {path}")
        print(json.dumps({"event": "e2_checkpoint_resume", "path": str(path.resolve()), "completed_folds": len(state["completed_folds"])}, ensure_ascii=False), flush=True)
        return path, state
    state = {
        "kind": CHECKPOINT_KIND,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "completed_folds": {},
        "complete": False,
    }
    write_json(path, state)
    print(json.dumps({"event": "e2_checkpoint_start", "path": str(path.resolve())}, ensure_ascii=False), flush=True)
    return path, state


def _checkpoint_key(regime: str, seed: int, fold: int) -> str:
    return f"{regime}|seed={seed}|fold={fold}"


def _restore_completed_fold(
    saved: Mapping[str, Any], validation_qids: Sequence[str], *, checkpoint_key: str,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    if saved.get("validation_qids") != list(validation_qids):
        raise V27ContractError(f"{checkpoint_key}: validation partition differs from checkpoint")
    predictions = saved.get("predictions")
    report = saved.get("fold_report")
    if not isinstance(predictions, dict) or not isinstance(report, dict):
        raise V27ContractError(f"{checkpoint_key}: incomplete saved fold")
    if set(predictions) != set(validation_qids):
        raise V27ContractError(f"{checkpoint_key}: saved predictions do not exactly cover the validation fold")
    return predictions, report


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _autocast(device: str):
    import torch

    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)


def _e1_manifest_contract(args: argparse.Namespace) -> dict[str, Any]:
    meta = _read_json(args.e1_manifest_meta)
    if meta.get("kind") != "legalir_e1_label_blind_evidence_frozen_ce":
        raise V27ContractError("E2 requires the sealed E1 label-blind manifest")
    if meta.get("reference_loaded") is not False or meta.get("holdout_opened") is not False:
        raise V27ContractError("E1 provenance does not attest label separation")
    if meta.get("stage1_sha256") != sha256_file(args.stage1):
        raise V27ContractError("E1 manifest Stage-1 provenance drift")
    if meta.get("stage2_sha256") != sha256_file(args.stage2):
        raise V27ContractError("E1 manifest Stage-2 provenance drift")
    if meta.get("anchor_sha256") != sha256_file(args.anchor):
        raise V27ContractError("E1 manifest anchor provenance drift")
    return meta


def model_checkpoint_fingerprint(model_path: Path) -> dict[str, str]:
    """Fingerprint model files, including weights, before a new E2 refit.

    E2's old reports recorded only ``config.json``.  E3-B derives a new model
    for every outer partition, so a same-path checkpoint replacement must be
    visible in its lineage contract as well.
    """

    result: dict[str, str] = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        path = Path(model_path) / name
        if path.is_file():
            result[name] = sha256_file(path)
    if "config.json" not in result or not {"model.safetensors", "pytorch_model.bin"} & set(result):
        raise FileNotFoundError(f"E2 reranker checkpoint is incomplete: {model_path}")
    return result


def load_e2_data(args: argparse.Namespace) -> E2Data:
    """Load the sealed complement universe and E2 supervision once.

    The caller is responsible for writing a label-blind manifest before this
    helper: loading the reference is intentionally unavoidable here because
    E2 fit and grouped OOF construction both require complement qrels.
    """

    for path, label in (
        (args.stage1, "stage1"), (args.stage2, "stage2"), (args.anchor, "anchor"),
        (args.reference, "reference"), (args.e1_manifest, "E1 manifest"),
        (args.e1_manifest_meta, "E1 metadata"), (args.e1_frozen_scores, "E1 scores"),
    ):
        _safe_complement_path(path, label)
        if not Path(path).is_file():
            raise FileNotFoundError(f"E2 input is missing: {path}")
    meta = _e1_manifest_contract(args)
    split = load_split("complement", args.stage1, args.stage2, args.anchor, args.reference)
    manifest = load_manifest(args.e1_manifest)
    _validate_manifest(split, manifest)
    frozen_scores = _load_frozen_scores(args.e1_frozen_scores)
    examples = _training_examples(split, manifest, frozen_scores, max_negatives=args.max_negatives)
    return E2Data(
        meta=meta,
        split=split,
        manifest=manifest,
        frozen_scores=frozen_scores,
        examples=examples,
        groups_by_regime=build_regime_groups(split, manifest),
    )


def _validate_manifest(split: Any, manifest: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
    expected = {
        (str(qid), str(doc_id))
        for qid, candidates in zip(split.qids, split.candidates, strict=True)
        for doc_id in candidates[:CANDIDATE_LIMIT]
    }
    if set(manifest) != expected:
        raise V27ContractError("E1 manifest does not exactly cover the frozen c35 universe")
    for key, row in manifest.items():
        blocks = row.get("blocks")
        if bool(row.get("empty_body")):
            if blocks:
                raise V27ContractError(f"{key}: empty document has E1 evidence")
        elif not isinstance(blocks, list) or not 1 <= len(blocks) <= 4:
            raise V27ContractError(f"{key}: E1 must supply 1..4 MIL instances")


def _load_frozen_scores(path: Path) -> dict[tuple[str, str], float | None]:
    """Read E1 scores, preserving the registered ``null`` EMPTY_BODY value."""

    result: dict[tuple[str, str], float | None] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Malformed E1 frozen score row {line_number}")
            key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
            value = row.get("frozen_ce_score")
            if not key[0] or not key[1] or key in result:
                raise ValueError(f"Invalid E1 frozen score row {line_number}")
            result[key] = None if value is None else float(value)
    return result


def _frozen_value(
    scores: Mapping[tuple[str, str], float | None],
    key: tuple[str, str],
    fallback: float,
) -> float:
    value = scores.get(key)
    return float(fallback if value is None else value)


def _training_examples(
    split: Any,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    frozen_scores: Mapping[tuple[str, str], float | None],
    *,
    max_negatives: int,
) -> dict[str, list[dict[str, Any]]]:
    """All qrels in c35 plus conservative, multi-source hard negatives."""

    if split.references is None or max_negatives <= 0:
        raise ValueError("E2 needs qrels and a positive negative cap")
    examples: dict[str, list[dict[str, Any]]] = {}
    for index, qid in enumerate(split.qids):
        qid = str(qid)
        c35 = [str(value) for value in split.candidates[index][:CANDIDATE_LIMIT]]
        gold = set(str(value) for value in split.references[index])
        positives = [doc_id for doc_id in c35 if doc_id in gold and not bool(manifest[(qid, doc_id)].get("empty_body"))]
        if not positives:
            # No in-universe positive provides no MIL localization signal.
            continue
        anchor = list(split.anchor[index])
        source_rank = {doc_id: rank for rank, doc_id in enumerate(split.source_candidates[index], 1)}
        ranked_sources = (
            anchor,
            sorted(c35, key=lambda doc_id: (-_frozen_value(frozen_scores, (qid, doc_id), -1e9), doc_id)),
            sorted(c35, key=lambda doc_id: (-float(split.neural_scores[index][doc_id]), doc_id)),
            sorted(c35, key=lambda doc_id: (source_rank.get(doc_id, 10_000), doc_id)),
        )
        negatives: list[str] = []
        for source in ranked_sources:
            for doc_id in source:
                if doc_id in gold or bool(manifest[(qid, doc_id)].get("empty_body")) or doc_id in negatives:
                    continue
                negatives.append(doc_id)
                if len(negatives) >= max_negatives:
                    break
            if len(negatives) >= max_negatives:
                break
        records: list[dict[str, Any]] = []
        rank = {doc_id: position for position, doc_id in enumerate(c35, 1)}
        for doc_id, label in [(doc_id, 1) for doc_id in positives] + [(doc_id, 0) for doc_id in negatives]:
            row = dict(manifest[(qid, doc_id)])
            row.update({
                "label": label,
                "candidate_rank": rank[doc_id],
                "anchor_member": doc_id in anchor,
                "frozen_ce_score": _frozen_value(frozen_scores, (qid, doc_id), -10.0),
            })
            records.append(row)
        examples[qid] = records
    return examples


def _encode_pairs(tokenizer: Any, pairs: Sequence[tuple[str, str]], device: str, max_length: int) -> dict[str, Any]:
    cache = getattr(tokenizer, "_e2_pair_cache", None)
    if cache is not None:
        return cache.encode(pairs, device)
    encoded = tokenizer([left for left, _ in pairs], [right for _, right in pairs], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _differentiable_scores(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, device: str, micro_batch: int) -> Any:
    import torch

    output: list[Any] = []
    for start in range(0, len(rows), max(1, micro_batch)):
        chunk = rows[start:start + max(1, micro_batch)]
        pairs: list[tuple[str, str]] = []
        counts: list[int] = []
        for row in chunk:
            passages = row_blocks(row)
            counts.append(len(passages))
            pairs.extend((str(row["query"]), passage) for passage in passages)
        if pairs:
            with _autocast(device):
                values = _raw_model_score(model(**_encode_pairs(tokenizer, pairs, device, E2_MAX_LENGTH))).float()
            cursor = 0
            for count in counts:
                if count:
                    chunk_scores = values[cursor:cursor + count]
                    cursor += count
                    output.append(E2_TEMPERATURE * torch.logsumexp(chunk_scores / E2_TEMPERATURE, dim=0))
                else:
                    output.append(torch.tensor(-10.0, device=device))
        else:
            output.extend(torch.tensor(-10.0, device=device) for _ in counts)
    return torch.stack(output).float()


def e2_loss(scores: Any, rows: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, float]]:
    """Relevance + top-5 boundary + light frozen-anchor score regularizer."""

    import torch
    import torch.nn.functional as F

    labels = torch.tensor([float(row["label"]) for row in rows], device=scores.device)
    relevance = F.binary_cross_entropy_with_logits(scores, labels)
    pair_losses: list[Any] = []
    positive = [index for index, row in enumerate(rows) if int(row["label"]) == 1]
    negative = [index for index, row in enumerate(rows) if int(row["label"]) == 0]
    for pindex in positive:
        for nindex in negative:
            p_rank = int(rows[pindex]["candidate_rank"])
            n_rank = int(rows[nindex]["candidate_rank"])
            # A wrong document around position five is more costly than a
            # distant candidate.  The rank is frozen c35 order, never qrel.
            boundary_weight = 1.0 / (1.0 + min(abs(p_rank - 5), abs(n_rank - 5)))
            pair_losses.append(boundary_weight * F.softplus(-(scores[pindex] - scores[nindex])))
    boundary = torch.stack(pair_losses).mean() if pair_losses else torch.zeros((), device=scores.device)
    anchor_indices = [index for index, row in enumerate(rows) if bool(row["anchor_member"])]
    if anchor_indices:
        target = torch.tensor([float(rows[index]["frozen_ce_score"]) for index in anchor_indices], device=scores.device)
        anchor = F.mse_loss(torch.tanh(scores[anchor_indices] / 8.0), torch.tanh(target / 8.0))
    else:
        anchor = torch.zeros((), device=scores.device)
    total = E2_LOSS_WEIGHTS["relevance"] * relevance + E2_LOSS_WEIGHTS["boundary"] * boundary + E2_LOSS_WEIGHTS["anchor"] * anchor
    return total, {"relevance": float(relevance.detach()), "boundary": float(boundary.detach()), "anchor": float(anchor.detach())}


def score_e2_rows_detailed(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Score E1 evidence rows and retain both passage and MIL document logits."""

    import torch

    pairs: list[tuple[str, str, str, int, str]] = []
    for row in rows:
        passages = row_blocks(row)
        if not passages and not bool(row.get("empty_body")):
            raise V27ContractError(
                f"{row.get('qid')}/{row.get('doc_id')}: non-empty E1 row has no scoreable passages"
            )
        pairs.extend(
            (str(row["query"]), str(row["qid"]), str(row["doc_id"]), passage_index, text)
            for passage_index, text in enumerate(passages)
        )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), max(1, batch_size)):
            batch = pairs[start:start + max(1, batch_size)]
            encoded = _encode_pairs(tokenizer, [(left, passage) for left, _, _, _, passage in batch], device, E2_MAX_LENGTH)
            values = _raw_model_score(model(**encoded)).detach().float().cpu().tolist()
            for (_, qid, doc_id, _, _), value in zip(batch, values, strict=True):
                grouped[(qid, doc_id)].append(float(value))
    return {
        key: {
            "document_mil_score": temperature_logsumexp(values, E2_TEMPERATURE),
            "passage_logits": list(values),
        }
        for key, values in grouped.items()
    }


def _score_rows(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, device: str, batch_size: int) -> dict[tuple[str, str], float]:
    detailed = score_e2_rows_detailed(model, tokenizer, rows, device=device, batch_size=batch_size)
    return {key: float(value["document_mil_score"]) for key, value in detailed.items()}


def score_full_c35(
    split: Any,
    indices: Sequence[int],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Export exactly one auditable score row for every document in c35.

    Empty-body rows are represented as a known ``null`` score.  Every other
    c35 document must have a MIL and its underlying passage logits; silently
    inserting a fallback score would defeat E3-B's artifact checks.
    """

    target_rows = [
        manifest[(str(split.qids[index]), str(doc_id))]
        for index in indices
        for doc_id in split.candidates[index][:CANDIDATE_LIMIT]
    ]
    detailed = score_e2_rows_detailed(model, tokenizer, target_rows, device=device, batch_size=batch_size)
    exported: list[dict[str, Any]] = []
    for index in indices:
        qid = str(split.qids[index])
        for candidate_rank, raw_doc_id in enumerate(split.candidates[index][:CANDIDATE_LIMIT], 1):
            doc_id = str(raw_doc_id)
            row = manifest[(qid, doc_id)]
            empty_body = bool(row.get("empty_body"))
            score = detailed.get((qid, doc_id))
            if empty_body:
                if score is not None:
                    raise V27ContractError(f"{qid}/{doc_id}: EMPTY_BODY unexpectedly received a neural score")
                exported.append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "candidate_rank": candidate_rank,
                    "empty_body": True,
                    "document_mil_score": None,
                    "passage_logits": [],
                })
                continue
            if score is None:
                raise V27ContractError(f"{qid}/{doc_id}: missing E2 MIL score in frozen c35")
            exported.append({
                "qid": qid,
                "doc_id": doc_id,
                "candidate_rank": candidate_rank,
                "empty_body": False,
                "document_mil_score": float(score["document_mil_score"]),
                "passage_logits": [float(value) for value in score["passage_logits"]],
            })
    expected = len(indices) * CANDIDATE_LIMIT
    if len(exported) != expected:
        raise V27ContractError(f"E2 full-score export has {len(exported)} rows, expected {expected}")
    return exported


def model_state_sha256(model: Any) -> str:
    """Hash loaded best weights without retaining a multi-gigabyte checkpoint."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _predictions(split: Any, manifest: Mapping[tuple[str, str], Mapping[str, Any]], scores: Mapping[tuple[str, str], float]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for index, qid in enumerate(split.qids):
        c35 = [str(value) for value in split.candidates[index][:CANDIDATE_LIMIT]]
        ranked = sorted(c35, key=lambda doc_id: (-float(scores.get((str(qid), doc_id), -1e9)), -float(split.neural_scores[index][doc_id]), doc_id))
        answers = [doc_id for doc_id in ranked if not bool(manifest[(str(qid), doc_id)].get("empty_body"))][:5]
        for position, doc_id in enumerate(split.anchor[index]):
            if bool(manifest[(str(qid), str(doc_id))].get("empty_body")):
                answers = [value for value in answers if value != str(doc_id)]
                answers.insert(position, str(doc_id))
                answers = answers[:5]
        if len(answers) != 5 or len(set(answers)) != 5:
            raise V27ContractError(f"{qid}: E2 produced invalid output")
        result[str(qid)] = {"answer": answers}
    return result


def _inner_split(qids: Sequence[str], groups: np.ndarray, seed: int) -> tuple[list[str], list[str]]:
    from sklearn.model_selection import GroupShuffleSplit

    if len(qids) < 20 or len(set(groups.tolist())) < 3:
        return list(qids), []
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(qids, groups=groups))
    return [qids[index] for index in fit], [qids[index] for index in validation]


def _train_fold(
    args: argparse.Namespace, examples: Mapping[str, Sequence[Mapping[str, Any]]], split: Any,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]], train_qids: Sequence[str], inner_qids: Sequence[str], seed: int,
    run_label: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    _set_seed(seed)
    model, tokenizer = load_relevance_model(
        args.reranker_model,
        args.device,
        train=True,
        fp16=args.device == "cuda",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    tokenizer._e2_pair_cache = PairEncodingCache(tokenizer, E2_MAX_LENGTH, max_entries=100_000, shared_store=OrderedDict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")
    best_state = None
    best_recall, stale, history = float("-inf"), 0, []
    fit_qids = [qid for qid in train_qids if examples.get(qid)]
    for epoch in range(args.epochs):
        shuffled = list(fit_qids)
        random.Random(seed + epoch).shuffle(shuffled)
        epoch_started = time.monotonic()
        print(json.dumps({"event": "e2_epoch_start", "run": run_label, "epoch": epoch + 1, "epochs": args.epochs, "trainable_queries": len(shuffled), "micro_batch": args.micro_batch, "gradient_checkpointing": bool(args.gradient_checkpointing)}, ensure_ascii=False), flush=True)
        optimizer.zero_grad(set_to_none=True)
        loss_rows: list[dict[str, float]] = []
        for position, qid in enumerate(shuffled, 1):
            model.train()
            total, pieces = e2_loss(_differentiable_scores(model, tokenizer, examples[qid], device=args.device, micro_batch=args.micro_batch), examples[qid])
            loss_rows.append(pieces)
            (scaler.scale(total / args.gradient_accumulation) if args.device == "cuda" else total / args.gradient_accumulation).backward()
            if position % args.gradient_accumulation == 0 or position == len(shuffled):
                if args.device == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if args.device == "cuda":
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if position % 100 == 0 or position == len(shuffled):
                elapsed = time.monotonic() - epoch_started
                eta_seconds = (elapsed / position) * (len(shuffled) - position)
                print(json.dumps({"event": "e2_train_progress", "run": run_label, "epoch": epoch + 1, "queries_done": position, "queries_total": len(shuffled), "elapsed_seconds": round(elapsed, 1), "eta_epoch_seconds": round(eta_seconds, 1)}, ensure_ascii=False), flush=True)
        inner_recall = None
        if inner_qids:
            inner_indices = [split.qids.index(qid) for qid in inner_qids]
            rows = [manifest[(str(split.qids[index]), str(doc_id))] for index in inner_indices for doc_id in split.candidates[index][:CANDIDATE_LIMIT]]
            inner_recall = float(transition_metrics(_subset(split, inner_indices), _predictions(_subset(split, inner_indices), manifest, _score_rows(model, tokenizer, rows, device=args.device, batch_size=args.score_batch_size)))["recall"])
            if inner_recall > best_recall + 1e-12:
                best_recall, stale = inner_recall, 0
                best_state = {name: copy.deepcopy(value.detach().cpu()) for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 1:
                    history.append({"epoch": epoch + 1, "inner_recall": inner_recall, "loss": _mean_loss(loss_rows)})
                    break
        history.append({"epoch": epoch + 1, "inner_recall": inner_recall, "loss": _mean_loss(loss_rows)})
        print(json.dumps({"event": "e2_epoch_complete", "run": run_label, "epoch": epoch + 1, "elapsed_seconds": round(time.monotonic() - epoch_started, 1), "inner_recall": inner_recall, "loss": history[-1]["loss"]}, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, {
        "epochs": len(history),
        "history": history,
        "best_inner_recall": None if best_recall == float("-inf") else best_recall,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "micro_batch": int(args.micro_batch),
    }


def fit_e2_model(
    args: argparse.Namespace,
    examples: Mapping[str, Sequence[Mapping[str, Any]]],
    split: Any,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    fit_qids: Sequence[str],
    stop_qids: Sequence[str],
    seed: int,
    run_label: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Public E2 fit interface for E3-B's A-fit/A-stop partitions.

    It deliberately delegates to the legacy E2 loop so the registered MIL
    aggregation, loss, optimiser, AMP and early-stopping behaviour do not
    drift between E2 and E3-B.
    """

    return _train_fold(
        args, examples, split, manifest, fit_qids, stop_qids, seed, run_label,
    )


def _memory_probe(args: argparse.Namespace, examples: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Exercise the exact training memory path once without producing OOF data."""

    if args.device != "cuda":
        raise ValueError("E2 memory probe requires --device cuda")
    import torch

    qid = next((candidate for candidate in sorted(examples) if examples[candidate]), None)
    if qid is None:
        raise V27ContractError("E2 memory probe found no trainable complement query")
    _set_seed(E2_DEFAULT_SEEDS[0])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = tokenizer = optimizer = scaler = None
    try:
        model, tokenizer = load_relevance_model(
            args.reranker_model,
            args.device,
            train=True,
            fp16=True,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        tokenizer._e2_pair_cache = PairEncodingCache(tokenizer, E2_MAX_LENGTH, max_entries=256, shared_store=OrderedDict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        total, pieces = e2_loss(
            _differentiable_scores(model, tokenizer, examples[qid], device=args.device, micro_batch=args.micro_batch),
            examples[qid],
        )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "kind": E2_KIND,
            "stage": "memory_probe",
            "fit": True,
            "qid": qid,
            "micro_batch": int(args.micro_batch),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes_after_step": int(free_bytes),
            "device_total_bytes": int(total_bytes),
            "loss": pieces,
            "holdout_opened": False,
            "public_labels_used": False,
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "E2 exact full-AdamW memory probe ran out of CUDA memory. "
            "Do not run the OOF stress gate on this GPU; use a >=24GB GPU or a separately preregistered low-memory variant."
        ) from exc
    finally:
        del model, tokenizer, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


def _mean_loss(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) if rows else 0.0 for key in E2_LOSS_WEIGHTS}


def _subset(split: Any, indices: Sequence[int]) -> Any:
    from v27_a_common import V27Split
    return V27Split(split.name, [split.qids[i] for i in indices], [split.questions[i] for i in indices], [split.candidates[i] for i in indices], [split.anchor[i] for i in indices], [split.neural_scores[i] for i in indices], [split.result_rows[i] for i in indices], [split.references[i] for i in indices] if split.references is not None else None, [split.source_candidates[i] for i in indices], [split.source_results[i] for i in indices])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.micro_batch <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("E2 training values must be positive")
    data = load_e2_data(args)
    meta = data.meta
    split = data.split
    manifest = data.manifest
    examples = data.examples
    groups_by_regime = data.groups_by_regime
    requested = list(dict.fromkeys(args.regimes))
    if any(regime not in groups_by_regime for regime in requested):
        raise ValueError(f"Unknown regime; choose from {sorted(groups_by_regime)}")
    if args.preflight:
        return {"kind": E2_KIND, "stage": "preflight", "queries": len(split.qids), "trainable_queries": len(examples), "regimes_available": sorted(groups_by_regime), "labels_loaded": True, "holdout_opened": False, "public_labels_used": False}
    if args.memory_probe:
        return _memory_probe(args, examples)
    checkpoint_path, checkpoint_state = _load_or_create_checkpoint(args)
    completed_folds: dict[str, Any] = checkpoint_state["completed_folds"]
    all_reports: dict[str, Any] = {}
    for regime in requested:
        folds = group_folds(groups_by_regime[regime])
        regime_seeds: list[dict[str, Any]] = []
        for seed in args.seeds:
            oof: dict[str, dict[str, list[str]]] = {}
            fold_reports: list[dict[str, Any]] = []
            for fold, (train_indices, validation_indices) in enumerate(folds):
                train_qids = [str(split.qids[i]) for i in train_indices]
                validation_qids = [str(split.qids[i]) for i in validation_indices]
                checkpoint_key = _checkpoint_key(regime, seed, fold)
                saved_fold = completed_folds.get(checkpoint_key)
                if saved_fold is not None:
                    saved_predictions, saved_report = _restore_completed_fold(
                        saved_fold, validation_qids, checkpoint_key=checkpoint_key,
                    )
                    oof.update(saved_predictions)
                    fold_reports.append(saved_report)
                    print(json.dumps({"event": "e2_fold_resumed", "run": f"regime={regime};seed={seed};fold={fold + 1}/{len(folds)}"}, ensure_ascii=False), flush=True)
                    continue
                inner_groups = groups_by_regime[regime][train_indices]
                fit_qids, inner_qids = _inner_split(train_qids, inner_groups, seed + fold)
                run_label = f"regime={regime};seed={seed};fold={fold + 1}/{len(folds)}"
                print(json.dumps({"event": "e2_fold_start", "run": run_label, "train_queries": len(train_qids), "fit_queries": len(fit_qids), "inner_queries": len(inner_qids), "validation_queries": len(validation_indices)}, ensure_ascii=False), flush=True)
                model, tokenizer, training = _train_fold(args, examples, split, manifest, fit_qids, inner_qids, seed + fold, run_label)
                target_rows = [manifest[(str(split.qids[i]), str(doc_id))] for i in validation_indices for doc_id in split.candidates[i][:CANDIDATE_LIMIT]]
                predictions = _predictions(_subset(split, validation_indices), manifest, _score_rows(model, tokenizer, target_rows, device=args.device, batch_size=args.score_batch_size))
                oof.update(predictions)
                fold_report = {"fold": fold, "train_qids": len(train_qids), "inner_qids": len(inner_qids), "validation_qids": len(validation_indices), "training": training, "validation": {key: value for key, value in transition_metrics(_subset(split, validation_indices), predictions).items() if key != "rows"}}
                fold_reports.append(fold_report)
                completed_folds[checkpoint_key] = {
                    "regime": regime,
                    "seed": seed,
                    "fold": fold,
                    "validation_qids": validation_qids,
                    "predictions": predictions,
                    "fold_report": fold_report,
                }
                write_json(checkpoint_path, checkpoint_state)
                print(json.dumps({"event": "e2_fold_complete", "run": run_label, "epochs_completed": training["epochs"]}, ensure_ascii=False), flush=True)
                del model, tokenizer
                gc.collect()
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
            ordered = {str(qid): oof[str(qid)] for qid in split.qids}
            regime_seeds.append({"seed": seed, "folds": fold_reports, "metrics": {key: value for key, value in transition_metrics(split, ordered).items() if key != "rows"}, "predictions": ordered})
        all_reports[regime] = regime_seeds
    checkpoint_state["complete"] = True
    write_json(checkpoint_path, checkpoint_state)
    return {"kind": E2_KIND, "schema_version": E2_SCHEMA_VERSION, "split": "complement", "status": "oof_only_not_promoted", "e1_manifest": str(args.e1_manifest.resolve()), "e1_manifest_sha256": sha256_file(args.e1_manifest), "e1_meta_sha256": sha256_file(args.e1_manifest_meta), "frozen_score_sha256": sha256_file(args.e1_frozen_scores), "model_config_sha256": sha256_file(args.reranker_model / "config.json"), "runtime": {"gradient_checkpointing": bool(args.gradient_checkpointing), "micro_batch": int(args.micro_batch), "score_batch_size": int(args.score_batch_size)}, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(completed_folds), "resumable": True}, "loss": {"weights": E2_LOSS_WEIGHTS, "passage_aggregation": "temperature_logsumexp", "temperature": E2_TEMPERATURE, "boundary": "positive-negative pairs weighted by frozen c35 distance to rank 5", "anchor": "tanh MSE against frozen E1 CE score for V16b-anchor documents"}, "examples": {"queries": len(examples), "documents": sum(len(rows) for rows in examples.values())}, "e1_provenance": meta, "regimes": all_reports, "holdout_opened": False, "public_labels_used": False, "promotion_allowed": False}


def main() -> int:
    args = parse_args()
    output = args.output_dir / "e2_mil_oof_memory_probe.json" if args.memory_probe else _report_path(args)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite E2 report: {output}")
    report = run(args)
    write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": report["status"] if "status" in report else report["stage"], "holdout_opened": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
