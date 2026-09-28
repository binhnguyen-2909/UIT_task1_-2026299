"""Validate only V36 condition/exception counterfactuals with Qwen as a teacher.

The teacher is a second opinion.  It cannot relabel qrels and its output is
never used as a final retrieval decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the candidate legal provision directly provides the legal basis needed to answer the question. "
    "Distinguish carefully: legal subject, regulated act, condition, exception, negation, authority, numerical threshold, time limit, and legal consequence. "
    "A document that is only topically similar but does not directly support the question is irrelevant."
)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = [row for row in read_jsonl(args.counterfactuals) if row.get("teacher_validation_required")]
    if not rows:
        raise RuntimeError("no teacher-validation rows")
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        str(args.model.resolve()),
        device=device,
        prompts={"legal": INSTRUCTION},
        default_prompt_name="legal",
        local_files_only=True,
        max_length=args.max_length,
        model_kwargs={"torch_dtype": torch.float16} if device.startswith("cuda") else {},
    )
    original_pairs = [(row["original_query"], row["premise"]) for row in rows]
    counterfactual_pairs = [(row["counterfactual_query"], row["premise"]) for row in rows]
    original = np.asarray(model.predict(original_pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True), dtype=np.float64)
    counterfactual = np.asarray(model.predict(counterfactual_pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True), dtype=np.float64)
    output = []
    for row, original_score, counterfactual_score in zip(rows, original, counterfactual, strict=True):
        margin = float(original_score - counterfactual_score)
        output.append({
            "counterfactual_id": row["counterfactual_id"],
            "qid": row["qid"],
            "factor": row["factor"],
            "original_query": row["original_query"],
            "counterfactual_query": row["counterfactual_query"],
            "gold_doc": row["gold_doc"],
            "original_teacher_score": float(original_score),
            "counterfactual_teacher_score": float(counterfactual_score),
            "teacher_margin_original_minus_counterfactual": margin,
            "teacher_supports_flip": bool(margin > 0.0),
            "teacher_is_second_opinion_only": True,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "v36_b_qwen_teacher_validation.jsonl"
    write_jsonl(output_path, output)
    margins = np.asarray([row["teacher_margin_original_minus_counterfactual"] for row in output], dtype=np.float64)
    report = {
        "kind": "legalir_v36_b_qwen_teacher_validation",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "training_performed": False,
        "role": "second_opinion_only_no_auto_relabel",
        "model": str(args.model.resolve()),
        "device": device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "rows": len(output),
        "teacher_supports_flip": int(sum(row["teacher_supports_flip"] for row in output)),
        "teacher_support_rate": float(np.mean([row["teacher_supports_flip"] for row in output])),
        "margin_median": float(np.median(margins)),
        "margin_mean": float(np.mean(margins)),
        "inputs": {"counterfactuals": {"path": str(args.counterfactuals.resolve()), "sha256": sha256_file(args.counterfactuals)}},
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v36_b_qwen_teacher_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
