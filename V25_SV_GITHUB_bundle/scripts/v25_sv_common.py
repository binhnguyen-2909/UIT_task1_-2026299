"""Shared, single-config data and evidence helpers for V25-SV."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


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


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("kind") != "legalir_v25_sv_resolved_config" or config.get("status") != "sealed_before_forward_pass":
        raise ValueError("V25-SV config is not sealed")
    if config.get("split") != "clean_complement_only" or config.get("holdout_opened") is not False or config.get("public_labels_used") is not False:
        raise ValueError("V25-SV config is not clean-complement-only")
    without_hash = dict(config)
    recorded = without_hash.pop("resolved_config_sha256", None)
    if recorded != json_hash(without_hash):
        raise ValueError("V25-SV resolved config self-hash mismatch")
    return config


def load_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(str(config["inputs"]["metadata_path"]))
    if sha256_file(path) != config["inputs"]["hashes"]["metadata"]:
        raise ValueError("metadata checksum differs from resolved config")
    payload = read_json(path)
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("V25-SV metadata kind/split mismatch")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("V25-SV metadata exposes external labels")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload.get("questions", []):
        qid = str(raw["qid"])
        anchor = [str(value) for value in raw["anchor_answer"]]
        c35 = [str(value) for value in raw["candidates_c35"]]
        references = {str(value) for value in raw["references"]}
        if len(anchor) != 5 or len(set(anchor)) != 5 or len(c35) != 35 or not references:
            raise ValueError(f"{qid}: metadata candidate/qrel shape invalid")
        result[qid] = {
            "qid": qid,
            "query": str(raw["query"]),
            "anchor": anchor,
            "c35": c35,
            "references": references,
            "anchor_scores": {doc: float(score) for doc, score in zip(c35, raw["anchor_scores"], strict=True)},
            "fold": int(raw.get("fold", -1)),
        }
    if not result:
        raise ValueError("metadata has no questions")
    return result


def recall(anchor: Sequence[str], pair: Sequence[str], references: set[str]) -> float:
    return len((set(anchor[:3]) | {str(pair[0]), str(pair[1])}) & references) / len(references)


def load_and_verify_edits(config: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    report_path = Path(str(config["inputs"]["v23_b0_report_path"]))
    edits_path = Path(str(config["inputs"]["v23_b0_edit_rows_path"]))
    if sha256_file(report_path) != config["inputs"]["hashes"]["v23_b0_report"] or sha256_file(edits_path) != config["inputs"]["hashes"]["v23_b0_edit_rows"]:
        raise ValueError("V23-B0 input checksum differs from resolved config")
    report = read_json(report_path)
    if report.get("promotion_gate", {}).get("passed") is not False or report.get("holdout_opened") is not False:
        raise ValueError("V25-SV requires failed V23-B0 with holdout closed")
    rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with edits_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in questions:
                raise ValueError(f"V23-B0 edit qid outside metadata at line {line_number}")
            pair = [str(value) for value in row["new_pair"]]
            anchor = questions[qid]["anchor"]
            refs = questions[qid]["references"]
            expected_after = recall(anchor, pair, refs)
            baseline = recall(anchor, [anchor[3], anchor[4]], refs)
            expected_delta = expected_after - baseline
            expected_outcome = "IMPROVE" if expected_delta > 1e-12 else "HARM" if expected_delta < -1e-12 else "TIE"
            if abs(float(row["delta_recall"]) - expected_delta) > 1e-12:
                raise ValueError(f"V23-B0 delta mismatch at line {line_number}")
            if {"improve": "IMPROVE", "harm": "HARM", "neutral": "TIE"}.get(str(row["outcome"])) != expected_outcome:
                raise ValueError(f"V23-B0 label mismatch at line {line_number}")
            if len(pair) != 2 or len(set(pair)) != 2 or any(doc not in questions[qid]["c35"] for doc in pair) or any(doc in anchor[:3] for doc in pair):
                raise ValueError(f"V23-B0 pair escapes c35/frozen top-3 at line {line_number}")
            edit_type = str(row["edit_type"])
            changed_slots = [int(value) for value in row["changed_slots"]]
            expected_slots = [slot for slot, (before, after) in zip((4, 5), zip(anchor[3:], pair, strict=True), strict=True) if before != after]
            if changed_slots != expected_slots or edit_type not in {"identity", "one_slot4", "one_slot5", "two_slot"}:
                raise ValueError(f"V23-B0 edit shape mismatch at line {line_number}")
            if edit_type == "identity" and pair != anchor[3:]:
                raise ValueError(f"V23-B0 identity mismatch at line {line_number}")
            rows_by_qid[qid].append({
                "qid": qid,
                "edit_id": str(row["edit_id"]),
                "edit_type": edit_type,
                "changed_slots": changed_slots,
                "new_pair": pair,
                "delta_recall": expected_delta,
                "outcome": expected_outcome,
                "baseline_recall": baseline,
            })
    if set(rows_by_qid) != set(questions) or any(not rows_by_qid[qid] for qid in questions):
        raise ValueError("V23-B0 edit qid coverage mismatch")
    expected_total = int(report["edit_space"]["total_edits"])
    if sum(len(rows) for rows in rows_by_qid.values()) != expected_total:
        raise ValueError("V23-B0 edit count mismatch")
    force_identity = {qid for qid, rows in rows_by_qid.items() if len(rows) == 1 and rows[0]["edit_type"] == "identity"}
    return dict(rows_by_qid), force_identity


def load_fold_manifest(config: Mapping[str, Any], qids: set[str]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    path = Path(str(config["inputs"]["fold_manifest_path"]))
    if sha256_file(path) != config["inputs"]["hashes"]["fold_manifest"]:
        raise ValueError("fold manifest checksum differs from resolved config")
    payload = read_json(path)
    if payload.get("kind") != "legalir_v21_group_crossfit_manifest" or payload.get("split") != "clean_complement_only" or payload.get("all_registered_regimes_zero_overlap") is not True:
        raise ValueError("fold manifest is not the audited grouped manifest")
    folds: dict[str, dict[str, int]] = {}
    groups: dict[str, dict[str, str]] = {}
    for regime in config["cross_fit"]["regimes"]:
        raw = payload["regimes"][regime]
        assignments = {str(key): str(value) for key, value in raw["group_assignments"].items()}
        fold_by_qid: dict[str, int] = {}
        for fold_raw in raw["folds"]:
            fold = int(fold_raw["fold"])
            train = {str(value) for value in fold_raw["train_qids"]}
            validation = {str(value) for value in fold_raw["validation_qids"]}
            if {assignments[qid] for qid in train} & {assignments[qid] for qid in validation}:
                raise ValueError(f"{regime}/{fold}: group leakage")
            for qid in validation:
                if qid in fold_by_qid:
                    raise ValueError(f"{regime}: duplicate validation qid")
                fold_by_qid[qid] = fold
        if set(assignments) != qids or set(fold_by_qid) != qids:
            raise ValueError(f"{regime}: fold/group coverage drift")
        folds[regime] = fold_by_qid
        groups[regime] = assignments
    return folds, groups


def iter_document_texts(db_path: Path, document_ids: set[str]) -> Iterator[tuple[str, str]]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        parts: dict[str, list[str]] = defaultdict(list)
        titles: dict[str, str] = {}
        query = "SELECT doc_id, title, passage FROM chunks ORDER BY rowid"
        for raw_doc, raw_title, raw_passage in connection.execute(query):
            doc_id = str(raw_doc)
            if doc_id not in document_ids:
                continue
            title = str(raw_title or "").strip()
            passage = str(raw_passage or "").strip()
            if title and doc_id not in titles:
                titles[doc_id] = title
            if passage:
                parts[doc_id].append(passage)
        for doc_id in sorted(document_ids):
            body = "\n".join(parts.get(doc_id, []))
            title = titles.get(doc_id, "")
            text = "\n".join(value for value in (title, body) if value).strip()
            yield doc_id, text
    finally:
        connection.close()


def make_blocks(tokenizer: Any, document_id: str, text: str, block_tokens: int, overlap_tokens: int) -> list[dict[str, Any]]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
    if not token_ids:
        return []
    step = block_tokens - overlap_tokens
    blocks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(token_ids):
        ids = list(token_ids[start:start + block_tokens])
        blocks.append({
            "doc_id": document_id,
            "block_id": f"{document_id}::v25block::{index:04d}",
            "block_index": index,
            "start_token": start,
            "end_token": start + len(ids),
            "token_count": len(ids),
            "text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "token_ids": ids,
        })
        index += 1
        if start + block_tokens >= len(token_ids):
            break
        start += step
    return blocks


def select_non_overlapping(blocks: Sequence[Mapping[str, Any]], scored: Mapping[str, float], retain: int) -> list[dict[str, Any]]:
    ordered = sorted(blocks, key=lambda row: (-float(scored[str(row["block_id"])]), str(row["block_id"])))
    selected: list[dict[str, Any]] = []
    for block in ordered:
        start, end = int(block["start_token"]), int(block["end_token"])
        if any(start < int(other["end_token"]) and int(other["start_token"]) < end for other in selected):
            continue
        selected.append(dict(block))
        if len(selected) >= retain:
            break
    return selected


def evidence_token_ids(selected_blocks: Sequence[Mapping[str, Any]], tokenizer: Any, per_block: int, total: int) -> list[int]:
    if not selected_blocks:
        raise ValueError("cannot create evidence from an empty block selection")
    if len(selected_blocks) == 1:
        ids = list(selected_blocks[0]["token_ids"][:total])
    else:
        ids = []
        for block in selected_blocks[:2]:
            ids.extend(list(block["token_ids"][:per_block]))
    if len(ids) > total:
        ids = ids[:total]
    return ids


def build_bge_pair_inputs(tokenizer: Any, pairs: Sequence[tuple[str, str]], prompt: str, max_length: int) -> dict[str, Any]:
    sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    inputs: list[dict[str, list[int]]] = []
    for query, passage in pairs:
        query_ids = tokenizer(f"A: {query}", add_special_tokens=False, max_length=max_length * 3 // 4, truncation=True)["input_ids"]
        passage_ids = tokenizer(f"B: {passage}", add_special_tokens=False, max_length=max_length, truncation=True)["input_ids"]
        # transformers' current GemmaTokenizer is a TokenizersBackend and no
        # longer exposes prepare_for_model.  The old BGE helper passed
        # add_special_tokens=False and truncation="only_second", whose exact
        # operation here is concatenation followed by truncating only the
        # second sequence from the right.
        first_ids = [int(tokenizer.bos_token_id)] + list(query_ids)
        second_ids = list(sep_ids) + list(passage_ids)
        second_ids = second_ids[: max(0, max_length - len(first_ids))]
        ids = first_ids + second_ids + sep_ids + prompt_ids
        inputs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return tokenizer.pad(inputs, padding=True, max_length=max_length + len(sep_ids) + len(prompt_ids), pad_to_multiple_of=8, return_tensors="pt")


def build_verifier_input_ids(tokenizer: Any, config: Mapping[str, Any], query: str, original: Sequence[Sequence[int]], proposed: Sequence[Sequence[int]]) -> list[int]:
    raw = config["raw_evidence"]
    control_budget = int(raw["verifier_control_token_budget"])
    query_budget = int(raw["verifier_query_token_budget"])
    instruction = "Classify the proposed legal top-five edit as exactly one label: IMPROVE, TIE, or HARM."
    structural = " QUERY: ORIGINAL_SLOT_4: ORIGINAL_SLOT_5: PROPOSED_SLOT_4: PROPOSED_SLOT_5: DECISION:"
    control_ids = tokenizer(instruction + structural, add_special_tokens=False)["input_ids"][:control_budget]
    query_ids = tokenizer(query, add_special_tokens=False)["input_ids"][:query_budget]
    ids = [int(tokenizer.bos_token_id)] + list(control_ids) + list(query_ids)
    for evidence in (*original, *proposed):
        ids.extend(int(value) for value in evidence[: int(raw["verifier_document_evidence_tokens"])])
    if len(ids) > int(raw["max_sequence_length"]):
        raise ValueError(f"verifier input exceeds fixed max length: {len(ids)}")
    return ids


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)
