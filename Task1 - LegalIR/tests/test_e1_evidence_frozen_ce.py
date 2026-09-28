"""CPU-only contract tests for E1 evidence selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e1_evidence_frozen_ce import (  # noqa: E402
    E1_PROPOSAL_LIMIT,
    E1_READ_LIMIT,
    E1Document,
    _rank_predictions,
    make_structural_token_blocks,
    propose_evidence_blocks,
    select_read_blocks,
)
from v27_a_common import EvidenceBlock, V27Document  # noqa: E402


class _UnusedTokenizer:
    def __call__(self, *_: object, **__: object) -> object:  # pragma: no cover - safety guard
        raise AssertionError("precomputed blocks should avoid tokenizer work")


class _ToyTokenizer:
    def __init__(self) -> None:
        self.by_token: dict[str, int] = {}
        self.by_id: dict[int, str] = {}

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        output = []
        for token in text.split():
            value = self.by_token.setdefault(token, len(self.by_token) + 1)
            self.by_id[value] = token
            output.append(value)
        return {"input_ids": output}

    def decode(self, values: list[int], **_: object) -> str:
        return " ".join(self.by_id[value] for value in values)


def _document() -> V27Document:
    return V27Document(
        doc_id="d1",
        title="Luật ví dụ",
        body="Nội dung pháp luật đầy đủ",
        link="https://example.test/luat-vi-du",
        source_family="example",
        duplicate_family="example",
    )


def _block(index: int, text: str) -> EvidenceBlock:
    return EvidenceBlock(
        block_id=f"d1::{index}",
        block_index=index,
        text=text,
        token_ids=(index + 1,),
        selector="test",
        selector_score=0.0,
    )


class EvidenceProposalTests(unittest.TestCase):
    def test_union_is_capped_and_label_blind(self) -> None:
        blocks = [
            _block(0, "Điều 12 quy định về hồ sơ đăng ký."),
            _block(1, "Khoản 1 nêu điều kiện cấp giấy phép."),
            _block(2, "Điểm a hướng dẫn thời hạn giải quyết."),
            _block(3, "Chương II quy định trách nhiệm tổ chức."),
            _block(4, "Phụ lục có biểu mẫu thực hiện thủ tục."),
        ]
        dense = {block.block_id: float(10 - block.block_index) for block in blocks}
        proposals = propose_evidence_blocks(
            "Điều 12 quy định hồ sơ cấp giấy phép như thế nào?",
            _document(),
            _UnusedTokenizer(),
            dense,
            precomputed_blocks=blocks,
        )
        self.assertLessEqual(len(proposals), E1_PROPOSAL_LIMIT)
        self.assertTrue(proposals)
        self.assertTrue(all("source_ranks" in row for row in proposals))
        self.assertTrue(any("bqb_dense" in row["source_ranks"] for row in proposals))
        self.assertFalse(any("label" in row or "answer" in row for row in proposals))

    def test_mmr_returns_unique_ordered_read_set(self) -> None:
        proposals = [
            {"block_id": "a", "block_index": 2, "text": "Điều 5 hồ sơ gồm đơn đề nghị và tài liệu.", "proposal_score": 3.0},
            {"block_id": "b", "block_index": 3, "text": "Điều 5 hồ sơ gồm đơn đề nghị và tài liệu bổ sung.", "proposal_score": 2.9},
            {"block_id": "c", "block_index": 0, "text": "Chương I quy định nguyên tắc chung.", "proposal_score": 2.0},
            {"block_id": "d", "block_index": 5, "text": "Khoản 2 nêu thời hạn xử lý là 15 ngày.", "proposal_score": 1.9},
            {"block_id": "e", "block_index": 7, "text": "Phụ lục kèm mẫu hồ sơ thủ tục.", "proposal_score": 1.8},
        ]
        selected = select_read_blocks(proposals)
        self.assertLessEqual(len(selected), E1_READ_LIMIT)
        self.assertEqual(len({row["block_id"] for row in selected}), len(selected))
        self.assertEqual([row["block_index"] for row in selected], sorted(row["block_index"] for row in selected))
        self.assertTrue(all("selection_score" in row for row in selected))

    def test_fourth_passage_is_gated_by_uncertainty(self) -> None:
        proposals = [
            {"block_id": "a", "block_index": 0, "text": "Điều 1 nội dung một.", "proposal_score": 3.0},
            {"block_id": "b", "block_index": 1, "text": "Điều 2 nội dung hai.", "proposal_score": 2.5},
            {"block_id": "c", "block_index": 2, "text": "Điều 3 nội dung ba.", "proposal_score": 2.0},
            {"block_id": "d", "block_index": 3, "text": "Phụ lục không liên quan.", "proposal_score": 0.1},
        ]
        self.assertEqual(len(select_read_blocks(proposals)), 3)

    def test_empty_document_cannot_emit_evidence(self) -> None:
        empty = V27Document("d0", "", "", "", "missing", "missing")
        self.assertEqual(
            propose_evidence_blocks("câu hỏi", empty, _UnusedTokenizer(), {}),
            [],
        )

    def test_raw_legal_chunking_repeats_parent_context(self) -> None:
        document = E1Document(
            "d2",
            "Luật kiểm tra",
            "Chương I\nĐiều 1. Quy định chung\n1. Nội dung khoản một.\nĐiều 2. Quy định khác\n1. Nội dung khoản hai.",
            "https://example.test/d2",
        )
        blocks, metadata = make_structural_token_blocks(
            _ToyTokenizer(), document, block_tokens=12, overlap_tokens=2
        )
        self.assertGreaterEqual(len(blocks), 2)
        self.assertEqual(set(block.block_id for block in blocks), set(metadata))
        self.assertTrue(all("[TITLE]" in block.text for block in blocks))
        self.assertTrue(any(value["article_number"] == "1" for value in metadata.values()))

    def test_empty_body_guard_keeps_only_anchor_empty_ids(self) -> None:
        candidates = [f"d{index}" for index in range(35)]
        split = SimpleNamespace(
            qids=["q1"],
            candidates=[candidates],
            neural_scores=[{doc_id: 0.0 for doc_id in candidates}],
            anchor=[["d0", "d1", "d2", "d3", "d4"]],
        )
        manifest = {
            ("q1", doc_id): {"empty_body": doc_id in {"d2", "d20"}}
            for doc_id in candidates
        }
        scores = {("q1", doc_id): float(100 - index) for index, doc_id in enumerate(candidates)}
        scores[("q1", "d20")] = 500.0
        answers = _rank_predictions(split, scores, manifest)["q1"]["answer"]
        self.assertIn("d2", answers)
        self.assertNotIn("d20", answers)


if __name__ == "__main__":
    unittest.main()
