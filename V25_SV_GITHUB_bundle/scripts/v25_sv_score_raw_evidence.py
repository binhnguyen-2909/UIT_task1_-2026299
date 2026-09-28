"""Score V25-SV raw blocks with the frozen BGE-Gemma checkpoint.

The model is loaded in BF16 exactly as resolved.  We use the underlying Gemma
body plus its unchanged LM head only at the final position; this is numerically
equivalent to the model-card ``logits[:, -1, yes_loc]`` score while avoiding the
unneeded vocab logits for every sequence position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v25_sv_common import (  # noqa: E402
    build_bge_pair_inputs,
    evidence_token_ids,
    load_and_verify_edits,
    load_config,
    load_metadata,
    select_non_overlapping,
    sha256_file,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iter_manifest(path: Path, expected_sha: str) -> Iterator[dict[str, Any]]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, encoded in enumerate(source, 1):
            digest.update(encoded)
            row = json.loads(encoded)
            if int(row["token_count"]) != len(row["token_ids"]):
                raise ValueError(f"raw block token count mismatch at line {line_number}")
            yield row
    actual = digest.hexdigest()
    if actual != expected_sha:
        raise ValueError(f"raw block manifest checksum differs from provenance: {actual} != {expected_sha}")


def load_model(config: dict[str, Any], device: torch.device) -> tuple[Any, Any, int]:
    model_dir = Path(str(config["model"]["local_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if sha256_file(model_dir / "tokenizer.json") != config["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer checksum differs from resolved config")
    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1:
        raise ValueError(f"Yes must be a single token for the sealed model, got {yes_ids}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V25-SV resolved inference_device=cuda but CUDA is unavailable")
    torch_dtype = getattr(torch, str(config["model"]["dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    actual_params = sum(int(parameter.numel()) for parameter in model.parameters())
    if actual_params != int(config["model"]["exact_parameter_count"]):
        raise ValueError(f"model parameter count mismatch: {actual_params}")
    model = model.to(device)
    model.eval()
    # The BGE model-card score is the final-position Yes logit.  Calling the
    # Gemma body and LM head at that one position produces the same value.
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise ValueError("resolved checkpoint does not expose the expected Gemma body/lm_head")
    return model, tokenizer, int(yes_ids[0])


def score_pairs(
    model: Any,
    tokenizer: Any,
    yes_loc: int,
    pairs: list[tuple[str, str]],
    prompt: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        inputs = build_bge_pair_inputs(tokenizer, pairs[start:start + batch_size], prompt, max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            body = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            last_index = inputs["attention_mask"].sum(dim=1).to(dtype=torch.long) - 1
            batch_index = torch.arange(body.last_hidden_state.shape[0], device=device)
            last_hidden = body.last_hidden_state[batch_index, last_index]
            logits = model.lm_head(last_hidden)
            scores.extend(logits[:, yes_loc].float().detach().cpu().tolist())
    if any(not torch.isfinite(torch.tensor(score)) for score in scores):
        raise FloatingPointError("non-finite BGE-Gemma raw evidence score")
    return [float(score) for score in scores]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--block-manifest", type=Path, required=True)
    parser.add_argument("--block-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if str(config["raw_evidence"]["inference_device"]) != "cuda":
        raise ValueError("V25-SV only permits the preregistered CUDA inference device")
    provenance = json.loads(args.block_provenance.read_text(encoding="utf-8"))
    if provenance.get("config_sha256") != config["resolved_config_sha256"] or provenance.get("holdout_opened") is not False or provenance.get("labels_written") is not False:
        raise ValueError("raw block provenance does not match the sealed label-free config")
    config_hash = str(config["resolved_config_sha256"])
    manifest_rows = iter_manifest(args.block_manifest, str(provenance["block_manifest_sha256"]))

    questions = load_metadata(config)
    rows_by_qid, force_identity = load_and_verify_edits(config, questions)
    qid_docs: dict[str, set[str]] = {}
    doc_qids: dict[str, set[str]] = defaultdict(set)
    for qid, rows in rows_by_qid.items():
        if qid in force_identity:
            continue
        docs = set(questions[qid]["anchor"][3:])
        for row in rows:
            docs.update(str(doc) for doc in row["new_pair"])
        qid_docs[qid] = docs
        for doc in docs:
            doc_qids[doc].add(qid)

    device = torch.device("cuda")
    model, tokenizer, yes_loc = load_model(config, device)
    prompt = str(config["raw_evidence"]["block_score_instruction"])
    max_length = int(config["raw_evidence"]["max_sequence_length"])
    score_batch_size = int(config["raw_evidence"]["block_scoring_batch_size"])
    if score_batch_size != 8:
        raise ValueError("V25-SV raw block scoring batch must remain the sealed value 8")

    if args.dry_run:
        first_doc = sorted(doc_qids)[0]
        selected_rows = []
        for row in manifest_rows:
            if str(row["doc_id"]) == first_doc:
                selected_rows.append(row)
                if len(selected_rows) >= 2:
                    break
        if len(selected_rows) < 2:
            raise ValueError("dry-run document has fewer than two raw blocks")
        first_qid = sorted(doc_qids[first_doc])[0]
        pairs = [(questions[first_qid]["query"], str(row["text"])) for row in selected_rows]
        first_scores = score_pairs(model, tokenizer, yes_loc, pairs, prompt, device, max_length, score_batch_size)
        second_scores = score_pairs(model, tokenizer, yes_loc, pairs, prompt, device, max_length, score_batch_size)
        if first_scores != second_scores:
            raise ValueError(f"non-deterministic BGE-Gemma dry-run scores: {first_scores} != {second_scores}")
        result = {
            "kind": "legalir_v25_sv_raw_score_dry_run",
            "config_sha256": config_hash,
            "model_revision": config["model"]["revision"],
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "yes_token_id": yes_loc,
            "qid": first_qid,
            "doc_id": first_doc,
            "block_ids": [str(row["block_id"]) for row in selected_rows],
            "scores": first_scores,
            "repeat_equal": True,
            "labels_loaded_for_features": False,
            "holdout_opened": False,
        }
        output = args.output_dir / "v25_sv_raw_score_dry_run.json"
        write_json(output, result)
        print(json.dumps({"dry_run": str(output.resolve()), "scores": first_scores, "device": str(device), "dtype": str(next(model.parameters()).dtype), "holdout_opened": False}, ensure_ascii=False))
        return 0

    # The manifest is sorted by document_id, so score and emit one document at
    # a time without retaining the full 1GB manifest in memory.
    pack_path = args.output_dir / "v25_sv_document_packs.jsonl"
    temporary = pack_path.with_name(pack_path.name + ".tmp")
    pack_digest = hashlib.sha256()
    pack_count = 0
    scored_pairs = 0
    seen_docs: set[str] = set()
    last_reported_docs = 0

    def emit_document(doc_id: str, blocks: list[dict[str, Any]]) -> None:
        nonlocal pack_count, scored_pairs
        if doc_id not in doc_qids:
            raise ValueError(f"raw manifest contains document outside candidate pool: {doc_id}")
        qids = sorted(doc_qids[doc_id])
        for qid in qids:
            pairs = [(questions[qid]["query"], str(block["text"])) for block in blocks]
            scores = score_pairs(model, tokenizer, yes_loc, pairs, prompt, device, max_length, score_batch_size)
            scored_pairs += len(scores)
            score_map = {str(block["block_id"]): float(score) for block, score in zip(blocks, scores, strict=True)}
            selected = select_non_overlapping(blocks, score_map, int(config["raw_evidence"]["retained_blocks_per_document"]))
            evidence = evidence_token_ids(
                selected,
                tokenizer,
                int(config["raw_evidence"]["tokens_per_retained_block"]),
                int(config["raw_evidence"]["tokens_per_document_evidence"]),
            )
            row = {
                "config_sha256": config_hash,
                "doc_id": doc_id,
                "evidence_token_ids": [int(value) for value in evidence],
                "qid": qid,
                "selected_blocks": [{"block_id": str(block["block_id"]), "block_index": int(block["block_index"]), "score": score_map[str(block["block_id"])]} for block in selected],
                "token_count": len(evidence),
            }
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            target.write(encoded)
            pack_digest.update(encoded)
            pack_count += 1

    current_doc = ""
    current_blocks: list[dict[str, Any]] = []
    with temporary.open("wb") as target:
        for row in manifest_rows:
            doc_id = str(row["doc_id"])
            if doc_id != current_doc:
                if current_doc:
                    emit_document(current_doc, current_blocks)
                    seen_docs.add(current_doc)
                current_doc = doc_id
                current_blocks = []
            current_blocks.append(row)
            if len(seen_docs) >= last_reported_docs + 100:
                last_reported_docs = (len(seen_docs) // 100) * 100
                print(json.dumps({"progress_documents": len(seen_docs), "packs": pack_count, "scored_pairs": scored_pairs}, ensure_ascii=False), flush=True)
        if current_doc:
            emit_document(current_doc, current_blocks)
            seen_docs.add(current_doc)
    if seen_docs != set(doc_qids):
        raise ValueError(f"scored raw evidence document coverage mismatch: missing={sorted(set(doc_qids) - seen_docs)[:10]}")
    os.replace(temporary, pack_path)
    pack_sha = pack_digest.hexdigest()
    result = {
        "kind": "legalir_v25_sv_document_packs",
        "config_sha256": config_hash,
        "model_revision": config["model"]["revision"],
        "yes_token_id": yes_loc,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "document_count": len(seen_docs),
        "pack_count": pack_count,
        "scored_query_block_pairs": scored_pairs,
        "pack_manifest_sha256": pack_sha,
        "labels_loaded_for_features": False,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    write_json(args.output_dir / "v25_sv_document_packs.provenance.json", result)
    print(json.dumps({"packs": str(pack_path.resolve()), "documents": len(seen_docs), "packs_count": pack_count, "scored_pairs": scored_pairs, "sha256": pack_sha, "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
