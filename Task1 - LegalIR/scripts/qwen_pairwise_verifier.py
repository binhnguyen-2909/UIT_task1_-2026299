"""Frozen pairwise verifier packaging. No model load here."""
from __future__ import annotations

import hashlib

PROMPT_TEMPLATE = (
    "Câu hỏi pháp luật (tiếng Việt):\n{question}\n\n"
    "Tài liệu A:\n{doc_a}\n\nTài liệu B:\n{doc_b}\n\n"
    "Chỉ chọn một: (A) Chỉ A phù hợp. (B) Chỉ B phù hợp. "
    "(C) Cả hai phù hợp. (D) Không đủ bằng chứng.\n"
    "Trả lời đúng một ký tự A/B/C/D."
)


def build_pair_prompt(question: str, doc_a_text: str, doc_b_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question.strip(), doc_a=doc_a_text.strip(), doc_b=doc_b_text.strip()
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def order_swap_consistent(score_ab: dict, score_ba: dict, threshold: float) -> bool:
    return bool(
        float(score_ab.get("challenger_margin", -1)) > float(threshold)
        and float(score_ba.get("challenger_margin", -1)) > float(threshold)
    )


def parse_four_choice(logits: list[float]) -> dict:
    import math

    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    probs = [e / s for e in exps]
    return {"A": probs[0], "B": probs[1], "both": probs[2], "neither": probs[3]}
