"""Local Task 2 evaluation aligned with the BTC ROUGE-L/METEOR scorer."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List


ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")


def btc_rouge_tokens(text: str) -> List[str]:
    """Mirror bundled rouge_score/tokenize.py: lowercase ASCII tokens only."""
    return ASCII_TOKEN_RE.findall(str(text).lower())


def lcs_length(left: List[str], right: List[str]) -> int:
    """LCS length using O(min(m,n)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, 1):
            if left_token == right_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_fmeasure(reference: str, prediction: str) -> float:
    reference_tokens = btc_rouge_tokens(reference)
    prediction_tokens = btc_rouge_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    if reference_tokens == prediction_tokens:
        return 1.0
    hits = lcs_length(reference_tokens, prediction_tokens)
    precision = hits / len(prediction_tokens)
    recall = hits / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def meteor(reference: str, prediction: str) -> float:
    """Call the same NLTK function and raw whitespace split as BTC."""
    reference_tokens = str(reference).split()
    prediction_tokens = str(prediction).split()
    # NLTK's exact-token case has one alignment chunk and can be evaluated
    # without WordNet. This keeps the copied warmup sanity check available in
    # a minimal environment while all non-trivial cases still require NLTK.
    if reference_tokens and reference_tokens == prediction_tokens:
        return 1.0 - 0.5 * (1.0 / len(reference_tokens)) ** 3
    try:
        from nltk.translate.meteor_score import meteor_score
        return float(meteor_score([reference_tokens], prediction_tokens))
    except ImportError as exc:
        raise RuntimeError("Install nltk from requirements-eval.txt for METEOR") from exc
    except LookupError as exc:
        raise RuntimeError(
            "NLTK wordnet/omw-1.4 are missing. Install them before the offline contest "
            "environment is locked; evaluation never downloads resources automatically."
        ) from exc


def _answers(data: Dict[str, Any], predictions: bool) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for qid, value in data.items():
        # Warmup files contain {question, answer}; the scoring container's
        # reference file contains the answer string directly. Support both.
        answer = value.get("answer") if isinstance(value, dict) else value
        if not isinstance(answer, str):
            raise ValueError(f"{qid}: answer must be a string")
        output[str(qid)] = answer
    return output


def evaluate_data(predictions: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, float]:
    predicted = _answers(predictions, predictions=True)
    truth = _answers(references, predictions=False)
    if set(predicted) != set(truth):
        missing = sorted(set(truth) - set(predicted))
        extra = sorted(set(predicted) - set(truth))
        raise ValueError(f"Question ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    rouge_scores = [rouge_l_fmeasure(truth[qid], predicted[qid]) for qid in predicted]
    meteor_scores = [meteor(truth[qid], predicted[qid]) for qid in predicted]
    count = len(predicted)
    return {
        "rouge": sum(rouge_scores) / count if count else 0.0,
        "meteor": sum(meteor_scores) / count if count else 0.0,
        "questions": count,
    }


def load_json_or_zip(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            if names != ["submission.json"]:
                raise ValueError(f"ZIP must contain exactly submission.json, found {names}")
            data = json.loads(archive.read("submission.json").decode("utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    scores = evaluate_data(load_json_or_zip(args.predictions), load_json_or_zip(args.references))
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
