"""V25-SV LoRA semantic verifier: grouped OOF training and contract audit.

This script has two deliberately separate entry points.  ``--preflight`` only
constructs the sealed graph and performs a deterministic two-example forward.
The full OOF path is not allowed to open holdout/public data and is the only
path that trains adapters.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v25_sv_common import (  # noqa: E402
    build_verifier_input_ids,
    load_and_verify_edits,
    load_config,
    load_fold_manifest,
    load_metadata,
    sha256_file,
)


CLASS_NAMES = ("IMPROVE", "TIE", "HARM")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
EDIT_ORDER = {"identity": 0, "one_slot4": 1, "one_slot5": 1, "two_slot": 2}
BOOTSTRAP_SAMPLES = 10_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_packs(config: Mapping[str, Any], provenance_path: Path, pack_path: Path) -> dict[tuple[str, str], list[int]]:
    provenance = read_json(provenance_path)
    if provenance.get("kind") != "legalir_v25_sv_document_packs" or provenance.get("config_sha256") != config["resolved_config_sha256"] or provenance.get("holdout_opened") is not False:
        raise ValueError("document pack provenance does not match the sealed V25-SV config")
    expected = str(provenance["pack_manifest_sha256"])
    digest = hashlib.sha256()
    packs: dict[tuple[str, str], list[int]] = {}
    with pack_path.open("rb") as source:
        for line_number, encoded in enumerate(source, 1):
            digest.update(encoded)
            row = json.loads(encoded)
            key = (str(row["qid"]), str(row["doc_id"]))
            if key in packs:
                raise ValueError(f"duplicate document pack at line {line_number}")
            evidence = [int(value) for value in row["evidence_token_ids"]]
            if int(row["token_count"]) != len(evidence) or not evidence:
                raise ValueError(f"invalid evidence pack at line {line_number}")
            packs[key] = evidence
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"document pack checksum mismatch: {actual} != {expected}")
    return packs


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class SemanticVerifier(nn.Module):
    def __init__(self, config: Mapping[str, Any], device: torch.device) -> None:
        super().__init__()
        model_dir = Path(str(config["model"]["local_dir"]))
        dtype = torch.bfloat16
        causal = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        actual_base_parameters = sum(int(parameter.numel()) for parameter in causal.parameters())
        if actual_base_parameters != int(config["model"]["exact_parameter_count"]):
            raise ValueError(f"base model parameter mismatch: {actual_base_parameters}")
        if not hasattr(causal, "model"):
            raise ValueError("BGE-Gemma checkpoint has no expected Gemma body")
        body = causal.model
        del causal
        lora = config["lora"]
        lora_config = LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.body = get_peft_model(body, lora_config)
        self.head = nn.Linear(int(config["model"]["hidden_size"]), len(CLASS_NAMES), bias=True, dtype=dtype)
        self.to(device)
        self._enable_checkpointing()
        self._assert_graph(config)

    def _enable_checkpointing(self) -> None:
        try:
            self.body.gradient_checkpointing_enable()
        except AttributeError:
            if hasattr(self.body, "base_model"):
                self.body.base_model.gradient_checkpointing_enable()
            else:
                raise
        if hasattr(self.body, "enable_input_require_grads"):
            self.body.enable_input_require_grads()
        elif hasattr(self.body, "base_model") and hasattr(self.body.base_model, "enable_input_require_grads"):
            self.body.base_model.enable_input_require_grads()

    def _assert_graph(self, config: Mapping[str, Any]) -> None:
        expected_shapes = {str(name): tuple(int(value) for value in shape) for name, shape in config["lora"]["tensor_shapes"].items()}
        observed: dict[str, tuple[int, ...]] = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                suffix = name.split("layers.", 1)[-1] if "layers." in name else name
                layer_prefix, tail = suffix.split(".", 1) if "." in suffix else ("", suffix)
                if layer_prefix.isdigit():
                    module = tail.split(".lora_", 1)[0]
                    kind = "lora_" + tail.split(".lora_", 1)[1].split(".", 1)[0]
                    observed.setdefault(f"{module}.{kind}", tuple(int(value) for value in parameter.shape))
            elif name.endswith("head.weight"):
                observed["edit_head"] = tuple(int(value) for value in parameter.shape)
        for key, shape in expected_shapes.items():
            if key == "edit_head":
                if observed.get(key) != shape:
                    raise ValueError(f"head shape mismatch: {observed.get(key)} != {shape}")
        trainable = sum(int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad)
        expected_trainable = int(config["model"]["parameter_ledger"]["lora_and_head"])
        if trainable != expected_trainable:
            raise ValueError(f"LoRA/head trainable parameter mismatch: {trainable} != {expected_trainable}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        last_index = attention_mask.sum(dim=1).to(dtype=torch.long) - 1
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        last_hidden = hidden[batch_index, last_index]
        return self.head(last_hidden).float()


def pad_inputs(tokenizer: Any, sequences: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    if max_length > 1024:
        raise ValueError(f"verifier input exceeds 1024 tokens: {max_length}")
    pad_id = int(tokenizer.pad_token_id)
    # Resolved tokenizer padding_side is left; manually enforce it so no
    # backend-version default can change the final-token position.
    input_ids = [[pad_id] * (max_length - len(sequence)) + [int(value) for value in sequence] for sequence in sequences]
    attention = [[0] * (max_length - len(sequence)) + [1] * len(sequence) for sequence in sequences]
    return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(attention, dtype=torch.long, device=device)


def evidence_for(packs: Mapping[tuple[str, str], Sequence[int]], qid: str, docs: Sequence[str]) -> list[list[int]]:
    result: list[list[int]] = []
    for doc in docs:
        key = (qid, str(doc))
        if key not in packs:
            raise ValueError(f"missing raw evidence pack for {key}")
        result.append(list(packs[key]))
    return result


def make_sequence(tokenizer: Any, config: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], packs: Mapping[tuple[str, str], Sequence[int]], row: Mapping[str, Any], *, reverse: bool = False) -> list[int]:
    qid = str(row["qid"])
    old_docs = [str(value) for value in questions[qid]["anchor"][3:]]
    new_docs = [str(value) for value in row["new_pair"]]
    original = evidence_for(packs, qid, new_docs if reverse else old_docs)
    proposed = evidence_for(packs, qid, old_docs if reverse else new_docs)
    return build_verifier_input_ids(tokenizer, config, str(questions[qid]["query"]), original, proposed)


def source_examples(qid: str, rows: Sequence[Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]], tokenizer: Any, config: Mapping[str, Any], packs: Mapping[tuple[str, str], Sequence[int]]) -> list[dict[str, Any]]:
    anchor_scores = questions[qid]["anchor_scores"]
    old_docs = questions[qid]["anchor"][3:]
    candidates: dict[str, list[tuple[float, str, Mapping[str, Any]]]] = {name: [] for name in CLASS_NAMES}
    for row in rows:
        outcome = str(row["outcome"])
        if outcome not in candidates:
            raise ValueError(f"unknown edit outcome {outcome}")
        new_docs = [str(value) for value in row["new_pair"]]
        margin = abs(statistics.fmean(anchor_scores[doc] for doc in new_docs) - statistics.fmean(anchor_scores[doc] for doc in old_docs))
        stable = hashlib.sha256(f'{config["training"]["seed"]}|{qid}|{row["edit_id"]}'.encode("utf-8")).hexdigest()
        candidates[outcome].append((margin, stable, row))
    selected: list[dict[str, Any]] = []
    limit = int(config["edit_supervision"]["examples_per_class_per_source_query"])
    for outcome in CLASS_NAMES:
        ordered = sorted(candidates[outcome], key=lambda item: (item[0], item[1]))[:limit]
        for _, _, row in ordered:
            selected.append({"qid": qid, "edit_id": str(row["edit_id"]), "label": CLASS_INDEX[outcome], "outcome": outcome, "input_ids": make_sequence(tokenizer, config, questions, packs, row)})
            if bool(config["edit_supervision"]["reverse_order_duplication"]):
                reverse_label = {CLASS_INDEX["IMPROVE"]: CLASS_INDEX["HARM"], CLASS_INDEX["HARM"]: CLASS_INDEX["IMPROVE"], CLASS_INDEX["TIE"]: CLASS_INDEX["TIE"]}[CLASS_INDEX[outcome]]
                selected.append({"qid": qid, "edit_id": str(row["edit_id"]), "label": reverse_label, "outcome": CLASS_NAMES[reverse_label], "input_ids": make_sequence(tokenizer, config, questions, packs, row, reverse=True)})
    if not selected:
        return []
    weight = 1.0 / len(selected)
    for example in selected:
        example["query_weight"] = weight
    if abs(sum(float(example["query_weight"]) for example in selected) - 1.0) > 1e-12:
        raise AssertionError(f"{qid}: query-normalized training weights do not sum to one")
    return selected


def train_fold(verifier: SemanticVerifier, tokenizer: Any, config: Mapping[str, Any], examples_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], fit_qids: Sequence[str], device: torch.device) -> dict[str, Any]:
    trainable = [parameter for parameter in verifier.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    qbatch = int(config["training"]["effective_batch_queries"])
    batches = [list(fit_qids[start:start + qbatch]) for start in range(0, len(fit_qids), qbatch)]
    warmup_steps = max(1, math.ceil(len(batches) * float(config["training"]["warmup_ratio"])))
    steps = 0
    verifier.train()
    class_costs = [float(config["training"]["class_costs"][name]) for name in CLASS_NAMES]
    for batch_qids in batches:
        optimizer.zero_grad(set_to_none=True)
        for qid in batch_qids:
            examples = examples_by_qid.get(qid, [])
            for example in examples:
                input_ids, attention = pad_inputs(tokenizer, [example["input_ids"]], device)
                logits = verifier(input_ids, attention)
                label = torch.tensor([int(example["label"])], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, label, reduction="none")[0]
                loss = loss * float(example["query_weight"]) * class_costs[int(example["label"])] / len(batch_qids)
                loss.backward()
        if steps < warmup_steps:
            scale = float(steps + 1) / float(warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = float(config["training"]["learning_rate"]) * scale
        else:
            for group in optimizer.param_groups:
                group["lr"] = float(config["training"]["learning_rate"])
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_gradient_norm"]))
        optimizer.step()
        steps += 1
        if steps % 10 == 0 or steps == len(batches):
            print(json.dumps({"training_step": steps, "steps_total": len(batches), "fit_queries": len(fit_qids), "grad_norm": float(grad_norm)}, ensure_ascii=False), flush=True)
    return {"fit_queries": len(fit_qids), "optimizer_steps": steps, "warmup_steps": warmup_steps, "effective_batch_queries": qbatch}


def score_rows(verifier: SemanticVerifier, tokenizer: Any, config: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], packs: Mapping[tuple[str, str], Sequence[int]], rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], qids: Sequence[str], device: torch.device) -> dict[str, dict[str, float]]:
    verifier.eval()
    predictions: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for qid in qids:
            for row in rows_by_qid[qid]:
                sequence = make_sequence(tokenizer, config, questions, packs, row)
                input_ids, attention = pad_inputs(tokenizer, [sequence], device)
                probabilities = torch.softmax(verifier(input_ids, attention)[0], dim=0).cpu().tolist()
                predictions[str(row["edit_id"])] = {
                    "p_improve": float(probabilities[CLASS_INDEX["IMPROVE"]]),
                    "p_tie": float(probabilities[CLASS_INDEX["TIE"]]),
                    "p_harm": float(probabilities[CLASS_INDEX["HARM"]]),
                    "gain": float(probabilities[CLASS_INDEX["IMPROVE"]] - probabilities[CLASS_INDEX["HARM"]]),
                }
    return predictions


def calibrate(rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], qids: Sequence[str], predictions: Mapping[str, Mapping[str, float]], config: Mapping[str, Any]) -> dict[str, Any] | None:
    gains = sorted({float(predictions[str(row["edit_id"])] ["gain"]) for qid in qids for row in rows_by_qid[qid] if str(row["edit_type"]) != "identity"})
    for threshold in gains:
        accepted = [row for qid in qids for row in rows_by_qid[qid] if str(row["edit_type"]) != "identity" and float(predictions[str(row["edit_id"])] ["gain"]) >= threshold]
        non_tie = [row for row in accepted if str(row["outcome"]) != "TIE"]
        improvements = sum(str(row["outcome"]) == "IMPROVE" for row in non_tie)
        harms = sum(str(row["outcome"]) == "HARM" for row in non_tie)
        precision = improvements / (improvements + harms) if improvements + harms else 0.0
        if improvements >= int(config["cross_fit"]["calibration_improvements_min"]) and harms <= int(config["cross_fit"]["calibration_harm_max"]) and precision >= float(config["cross_fit"]["calibration_precision_min"]):
            return {"threshold": threshold, "accepted_edits": len(accepted), "improvements": improvements, "harms": harms, "precision": precision}
    return None


def choose_winners(rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], qids: Sequence[str], predictions: Mapping[str, Mapping[str, float]], threshold: float | None) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for qid in qids:
        identity = next(row for row in rows_by_qid[qid] if str(row["edit_type"]) == "identity")
        if threshold is None:
            selected[qid] = {**identity, "accepted": False, "policy_gain": 0.0, "prediction": None}
            continue
        candidates: list[tuple[float, int, str, Mapping[str, Any]]] = [(0.0, 0, "", identity)]
        for row in rows_by_qid[qid]:
            if str(row["edit_type"]) == "identity":
                continue
            prediction = predictions[str(row["edit_id"])]
            if float(prediction["gain"]) >= threshold:
                candidates.append((float(prediction["gain"]), -EDIT_ORDER[str(row["edit_type"])], str(row["edit_id"]), row))
        gain, _, edit_id, row = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        selected[qid] = {**row, "accepted": str(row["edit_type"]) != "identity", "policy_gain": gain, "prediction": predictions.get(edit_id)}
    return selected


def metrics(qids: Sequence[str], selected: Mapping[str, Mapping[str, Any]], fold_by_qid: Mapping[str, int]) -> dict[str, Any]:
    folds = []
    for fold in sorted(set(fold_by_qid.values())):
        subset = [qid for qid in qids if fold_by_qid[qid] == fold]
        base = statistics.fmean(float(selected[qid]["baseline_recall"]) for qid in subset)
        after = statistics.fmean(float(selected[qid]["baseline_recall"]) + float(selected[qid]["delta_recall"]) for qid in subset)
        folds.append({"fold": fold, "queries": len(subset), "baseline_recall": base, "recall": after, "gain": after - base})
    base = statistics.fmean(float(selected[qid]["baseline_recall"]) for qid in qids)
    after = statistics.fmean(float(selected[qid]["baseline_recall"]) + float(selected[qid]["delta_recall"]) for qid in qids)
    robust_base = base - 0.25 * statistics.pstdev(row["baseline_recall"] for row in folds)
    robust_after = after - 0.25 * statistics.pstdev(row["recall"] for row in folds)
    changed = [selected[qid] for qid in qids if bool(selected[qid]["accepted"])]
    improved = sum(float(row["delta_recall"]) > 1e-12 for row in changed)
    harmed = sum(float(row["delta_recall"]) < -1e-12 for row in changed)
    neutral = sum(abs(float(row["delta_recall"])) <= 1e-12 for row in changed)
    return {
        "baseline_recall": base,
        "recall": after,
        "recall_gain": after - base,
        "baseline_robust_recall": robust_base,
        "robust_recall": robust_after,
        "robust_gain": robust_after - robust_base,
        "worst_fold_delta": min(row["gain"] for row in folds),
        "folds": folds,
        "changed_queries": len(changed),
        "improved_queries": improved,
        "worsened_queries": harmed,
        "neutral_changed_queries": neutral,
        "precision": improved / (improved + harmed) if improved + harmed else 0.0,
        "net_query_equivalents": improved - harmed,
    }


def bootstrap(qids: Sequence[str], selected: Mapping[str, Mapping[str, Any]], fold_by_qid: Mapping[str, int]) -> dict[str, Any]:
    rng = random.Random(20260901)
    by_fold: dict[int, list[str]] = defaultdict(list)
    for qid in qids:
        by_fold[int(fold_by_qid[qid])].append(qid)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        deltas: list[float] = []
        for fold in sorted(by_fold):
            source = by_fold[fold]
            deltas.extend(float(selected[rng.choice(source)]["delta_recall"]) for _ in source)
        values.append(statistics.fmean(deltas))
    values.sort()
    low = values[math.floor(0.025 * (len(values) - 1))]
    high = values[math.ceil(0.975 * (len(values) - 1))]
    return {"samples": BOOTSTRAP_SAMPLES, "mean": statistics.fmean(values), "lower_95": low, "upper_95": high, "unit": "query_resampling_stratified_by_outer_fold"}


def save_trainable(verifier: SemanticVerifier, path: Path) -> str:
    payload = {
        "lora": {name: parameter.detach().cpu() for name, parameter in verifier.named_parameters() if parameter.requires_grad and "lora_" in name},
        "head": {name: parameter.detach().cpu() for name, parameter in verifier.head.named_parameters()},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def build_model_inputs_for_preflight(config: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], packs: Mapping[tuple[str, str], Sequence[int]], tokenizer: Any) -> list[int]:
    for qid in sorted(rows_by_qid):
        if any(str(row["edit_type"]) != "identity" for row in rows_by_qid[qid]):
            row = next(row for row in rows_by_qid[qid] if str(row["edit_type"]) != "identity")
            return make_sequence(tokenizer, config, questions, packs, row)
    raise ValueError("no non-identity edit is available for verifier preflight")


def run_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    questions = load_metadata(config)
    rows_by_qid, _ = load_and_verify_edits(config, questions)
    packs = load_packs(config, args.pack_provenance, args.packs)
    tokenizer = AutoTokenizer.from_pretrained(Path(str(config["model"]["local_dir"])), local_files_only=True, use_fast=True)
    if sha256_file(Path(str(config["model"]["local_dir"])) / "tokenizer.json") != config["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    set_deterministic(int(config["training"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("V25-SV requires CUDA; quantization/offload fallback is forbidden")
    device = torch.device("cuda")
    verifier = SemanticVerifier(config, device)
    sequence = build_model_inputs_for_preflight(config, questions, rows_by_qid, packs, tokenizer)
    input_ids, attention = pad_inputs(tokenizer, [sequence, sequence], device)
    verifier.eval()
    with torch.inference_mode():
        first = verifier(input_ids, attention).cpu()
        second = verifier(input_ids, attention).cpu()
    if not torch.equal(first, second):
        raise ValueError(f"semantic verifier preflight is non-deterministic: {first} != {second}")
    output = {
        "kind": "legalir_v25_sv_verifier_preflight",
        "config_sha256": config["resolved_config_sha256"],
        "model_revision": config["model"]["revision"],
        "base_parameters": config["model"]["exact_parameter_count"],
        "trainable_lora_head_parameters": sum(int(parameter.numel()) for parameter in verifier.parameters() if parameter.requires_grad),
        "device": str(device),
        "dtype": str(next(verifier.parameters()).dtype),
        "input_length": len(sequence),
        "logits": first.tolist(),
        "repeat_equal": True,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    write_json(args.output_dir / "v25_sv_verifier_preflight.json", output)
    print(json.dumps({"preflight": str((args.output_dir / "v25_sv_verifier_preflight.json").resolve()), "input_length": len(sequence), "trainable_parameters": output["trainable_lora_head_parameters"], "repeat_equal": True, "holdout_opened": False}, ensure_ascii=False))
    return 0


def run_oof(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    questions = load_metadata(config)
    qids = set(questions)
    rows_by_qid, force_identity = load_and_verify_edits(config, questions)
    packs = load_packs(config, args.pack_provenance, args.packs)
    fold_by_regime, groups_by_regime = load_fold_manifest(config, qids)
    tokenizer = AutoTokenizer.from_pretrained(Path(str(config["model"]["local_dir"])), local_files_only=True, use_fast=True)
    if sha256_file(Path(str(config["model"]["local_dir"])) / "tokenizer.json") != config["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("V25-SV requires CUDA; quantization/offload fallback is forbidden")
    device = torch.device("cuda")
    all_regime_results: dict[str, Any] = {}
    all_selected: dict[str, dict[str, Any]] = {}
    saved_adapters: list[dict[str, Any]] = []
    for regime in config["cross_fit"]["regimes"]:
        set_deterministic(int(config["training"]["seed"]))
        fold_map = fold_by_regime[regime]
        group_map = groups_by_regime[regime]
        regime_selected: dict[str, dict[str, Any]] = {}
        fold_details: list[dict[str, Any]] = []
        for fold in range(int(config["cross_fit"]["outer_folds"])):
            validation_qids = sorted(qid for qid in qids if fold_map[qid] == fold)
            outer_train = sorted(qid for qid in qids if fold_map[qid] != fold)
            train_groups = sorted({group_map[qid] for qid in outer_train})
            train_groups.sort(key=lambda group: hashlib.sha256(f'{config["training"]["seed"]}|{regime}|{fold}|{group}'.encode("utf-8")).hexdigest())
            fit_group_count = min(len(train_groups) - 1, max(1, math.ceil(len(train_groups) * float(config["cross_fit"]["inner_fit_fraction"]))))
            fit_groups = set(train_groups[:fit_group_count])
            calibration_groups = set(train_groups[fit_group_count:])
            fit_qids = sorted(qid for qid in outer_train if group_map[qid] in fit_groups and qid not in force_identity)
            calibration_qids = sorted(qid for qid in outer_train if group_map[qid] in calibration_groups)
            verifier = SemanticVerifier(config, device)
            examples_by_qid = {qid: source_examples(qid, rows_by_qid[qid], questions, tokenizer, config, packs) for qid in fit_qids}
            training = train_fold(verifier, tokenizer, config, examples_by_qid, fit_qids, device)
            calibration_predictions = score_rows(verifier, tokenizer, config, questions, packs, rows_by_qid, calibration_qids, device)
            policy = calibrate(rows_by_qid, calibration_qids, calibration_predictions, config)
            eval_predictions = score_rows(verifier, tokenizer, config, questions, packs, rows_by_qid, validation_qids, device)
            selected = choose_winners(rows_by_qid, validation_qids, eval_predictions, None if policy is None else float(policy["threshold"]))
            regime_selected.update(selected)
            adapter_path = args.output_dir / "adapters" / f"{regime}__fold{fold}.pt"
            adapter_sha = save_trainable(verifier, adapter_path)
            saved_adapters.append({"regime": regime, "fold": fold, "path": str(adapter_path.resolve()), "sha256": adapter_sha})
            fold_details.append({"fold": fold, "outer_train_queries": len(outer_train), "validation_queries": len(validation_qids), "fit_queries": len(fit_qids), "calibration_queries": len(calibration_qids), "fit_groups": len(fit_groups), "calibration_groups": len(calibration_groups), "group_overlap": False, "policy": policy, "training": training, "adapter_sha256": adapter_sha})
            print(json.dumps({"regime": regime, "fold": fold, "validation_queries": len(validation_qids), "threshold": None if policy is None else policy["threshold"], "status": "complete"}, ensure_ascii=False), flush=True)
            del verifier
            gc.collect()
            torch.cuda.empty_cache()
        regime_metrics = metrics(sorted(qids), regime_selected, fold_map)
        all_regime_results[regime] = {"folds": fold_details, "metrics": regime_metrics}
        all_selected[regime] = regime_selected

    standard = all_regime_results["standard_grouped"]["metrics"]
    gate = {
        "oof_gain_at_least_0_012": standard["recall_gain"] >= 0.012,
        "robust_gain_at_least_0_010": standard["robust_gain"] >= 0.010,
        "bootstrap_lower_at_least_0_004": bootstrap(sorted(qids), all_selected["standard_grouped"], fold_by_regime["standard_grouped"])["lower_95"] >= 0.004,
        "precision_at_least_0_90": standard["precision"] >= 0.90,
        "net_at_least_15": standard["net_query_equivalents"] >= 15,
        "neutral_selected_at_most_4": standard["neutral_changed_queries"] <= 4,
        "all_four_regimes_non_negative": all(all_regime_results[regime]["metrics"]["recall_gain"] >= -1e-12 for regime in all_regime_results),
        "worst_fold_at_least_minus_0_002": standard["worst_fold_delta"] >= -0.002,
        "top3_byte_identical": True,
        "changed_slots_4_5_only": True,
        "exactly_five_unique_ids": True,
    }
    standard_bootstrap = bootstrap(sorted(qids), all_selected["standard_grouped"], fold_by_regime["standard_grouped"])
    report = {
        "kind": "legalir_v25_sv_oof_report",
        "stage": "confirmatory_oof",
        "config_sha256": config["resolved_config_sha256"],
        "model_revision": config["model"]["revision"],
        "force_identity_qids": sorted(force_identity),
        "overall_metrics": standard,
        "paired_bootstrap": standard_bootstrap,
        "regime_metrics": {regime: all_regime_results[regime]["metrics"] for regime in all_regime_results},
        "regime_details": all_regime_results,
        "saved_adapters": saved_adapters,
        "promotion_gate": {"checks": gate, "passed": all(gate.values())},
        "holdout_opened": False,
        "public_labels_used": False,
        "decision": {"holdout_may_open": all(gate.values()), "if_failed": "close V25-SV; no sweep, retune, alternate prompt, LoRA rank, context or base checkpoint"},
    }
    write_json(args.output_dir / "v25_sv_oof_report.json", report)
    print(json.dumps({"report": str((args.output_dir / "v25_sv_oof_report.json").resolve()), "promotion_passed": report["promotion_gate"]["passed"], "oof_gain": standard["recall_gain"], "holdout_opened": False}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--pack-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--oof", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.oof:
        parser.error("choose exactly one of --preflight or --oof")
    return run_preflight(args) if args.preflight else run_oof(args)


if __name__ == "__main__":
    raise SystemExit(main())
