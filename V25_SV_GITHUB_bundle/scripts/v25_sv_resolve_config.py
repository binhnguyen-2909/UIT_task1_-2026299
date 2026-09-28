"""Resolve and seal the single V25-SV configuration before any forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_REVISION = "1787044f8b6fb740a9de4557c3a12377f84d9e17"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
FORBIDDEN_LABEL_FILES = ("train.json", "public-official.json", "holdout")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(model_dir: Path) -> int:
    from safetensors import safe_open

    index = read_json(model_dir / "model.safetensors.index.json")
    shards = sorted(set(str(value) for value in index["weight_map"].values()))
    total = 0
    seen: set[str] = set()
    for shard in shards:
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    continue
                shape = tuple(int(value) for value in handle.get_slice(name).get_shape())
                count = 1
                for value in shape:
                    count *= value
                total += count
                seen.add(name)
    return total


def anchor_and_qrel_hashes(metadata: Mapping[str, Any]) -> tuple[str, str]:
    anchors: dict[str, Any] = {}
    qrels: dict[str, Any] = {}
    for raw in metadata["questions"]:
        qid = str(raw["qid"])
        anchors[qid] = [str(value) for value in raw["anchor_answer"]]
        qrels[qid] = sorted(str(value) for value in raw["references"])
    return json_hash(anchors), json_hash(qrels)


def lora_shapes(config: Mapping[str, Any], rank: int) -> dict[str, list[int]]:
    hidden = int(config["hidden_size"])
    kv = hidden // int(config["num_attention_heads"])
    return {
        "q_proj.lora_A": [rank, hidden],
        "q_proj.lora_B": [hidden, rank],
        "k_proj.lora_A": [rank, hidden],
        "k_proj.lora_B": [kv, rank],
        "v_proj.lora_A": [rank, hidden],
        "v_proj.lora_B": [kv, rank],
        "o_proj.lora_A": [rank, hidden],
        "o_proj.lora_B": [hidden, rank],
        "edit_head": [3, hidden],
    }


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    prereg = Path(args.prereg).resolve()
    model_dir = Path(args.model_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    v23_report = Path(args.v23_b0_report).resolve()
    v23_edits = Path(args.v23_b0_edits).resolve()
    fold_manifest = Path(args.fold_manifest).resolve()
    contexts_db = Path(args.contexts_db).resolve()
    output_dir = Path(args.output_dir).resolve()
    required_model_files = [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
    ]
    if not all((model_dir / name).is_file() for name in required_model_files):
        raise FileNotFoundError("resolved V25 model directory is incomplete")
    for path, label in ((metadata_path, "metadata"), (v23_report, "V23-B0 report"), (v23_edits, "V23-B0 edits"), (fold_manifest, "fold manifest"), (contexts_db, "contexts database")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for token in FORBIDDEN_LABEL_FILES:
        if token in str(v23_report).casefold() or token in str(v23_edits).casefold():
            raise ValueError("V25-SV input path points to a forbidden external-label artifact")
    metadata = read_json(metadata_path)
    if metadata.get("split") != "clean_complement_only" or metadata.get("holdout_opened") is not False or metadata.get("public_labels_used") is not False:
        raise ValueError("V25-SV metadata must be clean-complement-only")
    v23 = read_json(v23_report)
    if v23.get("promotion_gate", {}).get("passed") is not False or v23.get("holdout_opened") is not False:
        raise ValueError("V25-SV requires the failed V23-B0 report with holdout closed")
    model_config = read_json(model_dir / "config.json")
    if model_config.get("model_type") != "gemma" or model_config.get("architectures") != ["GemmaForCausalLM"]:
        raise ValueError("resolved model is not the expected BGE-Gemma GemmaForCausalLM graph")
    exact_parameter_count = count_parameters(model_dir)
    hidden = int(model_config["hidden_size"])
    layers = int(model_config["num_hidden_layers"])
    rank = 32
    lora_shape_map = lora_shapes(model_config, rank)
    def shape_size(shape: Sequence[int]) -> int:
        size = 1
        for value in shape:
            size *= int(value)
        return size

    lora_parameters = sum(shape_size(shape) for name, shape in lora_shape_map.items() if name != "edit_head")
    lora_parameters = lora_parameters * layers + shape_size(lora_shape_map["edit_head"])
    anchor_hash, qrel_hash = anchor_and_qrel_hashes(metadata)
    model_hashes = {name: sha256_file(model_dir / name) for name in required_model_files if name.endswith((".json", ".model", ".safetensors"))}
    input_hashes = {
        "preregistration": sha256_file(prereg),
        "metadata": sha256_file(metadata_path),
        "v23_b0_report": sha256_file(v23_report),
        "v23_b0_edit_rows": sha256_file(v23_edits),
        "fold_manifest": sha256_file(fold_manifest),
        "contexts_database": sha256_file(contexts_db),
    }
    payload = {
        "kind": "legalir_v25_sv_resolved_config",
        "schema_version": 1,
        "status": "sealed_before_forward_pass",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": {
            "repository": "BAAI/bge-reranker-v2-gemma",
            "revision": MODEL_REVISION,
            "local_dir": str(model_dir),
            "architecture": "GemmaForCausalLM",
            "dtype": "bfloat16",
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "exact_parameter_count": exact_parameter_count,
            "parameter_ledger": {
                "v16b": 1090010000,
                "bge_gemma": exact_parameter_count,
                "lora_and_head": lora_parameters,
                "worst_case_total": 1090010000 + exact_parameter_count + lora_parameters,
                "under_4b": 1090010000 + exact_parameter_count + lora_parameters < 4_000_000_000,
            },
        },
        "tokenizer": {
            "revision": MODEL_REVISION,
            "tokenizer_json_sha256": sha256_file(model_dir / "tokenizer.json"),
            "tokenizer_model_sha256": sha256_file(model_dir / "tokenizer.model"),
            "padding_side": "left",
        },
        "lora": {
            "rank": rank,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": list(TARGET_MODULES),
            "tensor_shapes": lora_shape_map,
        },
        "raw_evidence": {
            "block_tokens": 192,
            "block_overlap_tokens": 48,
            "retained_blocks_per_document": 2,
            "tokens_per_document_evidence": 192,
            "tokens_per_retained_block": 96,
            "max_sequence_length": 1024,
            "block_scoring_batch_size": 8,
            "verifier_eval_batch_size": 1,
            "inference_device": "cuda",
            "verifier_control_token_budget": 63,
            "verifier_query_token_budget": 192,
            "verifier_document_evidence_tokens": 192,
            "instruction_tokens_max": 64,
            "query_tokens_max": 192,
            "original_evidence_tokens_max": 384,
            "proposed_evidence_tokens_max": 384,
            "block_score_instruction": "Với câu hỏi pháp lý A và đoạn bằng chứng pháp lý B, hãy xác định B có chứa câu trả lời cho A hay không. Chỉ trả lời 'Yes' hoặc 'No'.",
            "tie_break": "earliest block ID",
            "truncation": "from the end of the lowest-scoring retained block; query is never truncated first",
        },
        "edit_supervision": {
            "classes": ["IMPROVE", "TIE", "HARM"],
            "examples_per_class_per_source_query": 2,
            "reverse_order_duplication": True,
            "rank_margin": "absolute mean V16b anchor-score change over mutable slots",
            "missing_class_backfill": False,
            "query_weight_after_reversal": 1.0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.00002,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "effective_batch_queries": 32,
            "max_gradient_norm": 1.0,
            "seed": 20260901,
            "precision": "bf16",
            "checkpoint_selection": "final_step_only",
            "class_costs": {"IMPROVE": 1.0, "TIE": 0.5, "HARM": 2.0},
        },
        "cross_fit": {
            "regimes": ["standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood"],
            "outer_folds": 5,
            "inner_fit_fraction": 0.80,
            "inner_split": "group_hash",
            "calibration_precision_min": 0.90,
            "calibration_improvements_min": 5,
            "calibration_harm_max": 1,
            "gain_score": "P(IMPROVE)-P(HARM)",
            "no_regime_vote_or_veto": True,
        },
        "inputs": {
            "preregistration_path": str(prereg),
            "metadata_path": str(metadata_path),
            "v23_b0_report_path": str(v23_report),
            "v23_b0_edit_rows_path": str(v23_edits),
            "fold_manifest_path": str(fold_manifest),
            "contexts_database_path": str(contexts_db),
            "anchor_checksum": anchor_hash,
            "qrel_checksum": qrel_hash,
            "hashes": input_hashes,
            "model_file_hashes": model_hashes,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "deterministic_algorithms": True,
        },
        "contract": {
            "oof_gain_min": 0.012,
            "robust_gain_min": 0.010,
            "bootstrap_lower_min": 0.004,
            "precision_min": 0.90,
            "net_query_equivalents_min": 15,
            "neutral_selected_max": 4,
            "four_regimes_non_negative": True,
            "worst_fold_min": -0.002,
            "top3_byte_identical": True,
            "mutable_slots": [4, 5],
        },
    }
    payload["resolved_config_sha256"] = json_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "RESOLVED_CONFIG.json"
    temporary = config_path.with_name(config_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, config_path)
    (output_dir / "RESOLVED_CONFIG.sha256").write_text(sha256_file(config_path) + "  RESOLVED_CONFIG.json\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--v23-b0-report", type=Path, required=True)
    parser.add_argument("--v23-b0-edits", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--contexts-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = resolve(args)
    print(json.dumps({"config": str((args.output_dir / "RESOLVED_CONFIG.json").resolve()), "revision": config["model"]["revision"], "base_parameters": config["model"]["exact_parameter_count"], "total_parameters": config["model"]["parameter_ledger"]["worst_case_total"], "under_4b": config["model"]["parameter_ledger"]["under_4b"], "status": config["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
