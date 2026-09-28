"""V33-A0/A1 native multi-view consensus and RRF audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
V27_DIR = SCRIPT_DIR / "v27"
if str(V27_DIR) not in sys.path:
    sys.path.insert(0, str(V27_DIR))
from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402


def _num(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _load_views(path: Path, qids: set[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in qids:
                    result[qid] = {view: row[view] for view in ("Q0", "Q1", "Q2")}
    if set(result) != qids:
        raise ValueError(f"view coverage mismatch: {len(result)} != {len(qids)}")
    return result


def _rrf(ranks: dict[str, int]) -> float:
    return sum(1.0 / (60.0 + rank) for rank in ranks.values())


def _candidate(qviews: dict[str, list[dict[str, Any]]], anchor: list[str], policy: str) -> tuple[str | None, dict[str, Any]]:
    ranks = {view: {str(item["doc_id"]): int(item["rank"]) for item in rows} for view, rows in qviews.items()}
    if policy == "top1_vote2":
        k = 1
        candidates = {doc for view in ranks for doc, rank in ranks[view].items() if rank <= k}
        votes = {doc: sum(rank_map.get(doc, 10**9) <= k for rank_map in ranks.values()) for doc in candidates}
        eligible = [doc for doc in candidates if votes[doc] >= 2 and doc not in set(anchor[:4])]
    elif policy.startswith("top") and policy.endswith("_vote2"):
        k = int(policy[3:-6])
        candidates = {doc for view in ranks for doc, rank in ranks[view].items() if rank <= k}
        votes = {doc: sum(rank_map.get(doc, 10**9) <= k for rank_map in ranks.values()) for doc in candidates}
        eligible = [doc for doc in candidates if votes[doc] >= 2 and doc not in set(anchor[:4])]
    elif policy == "rrf_top20":
        k = 20
        candidates = {doc for view in ranks for doc, rank in ranks[view].items() if rank <= k}
        votes = {doc: sum(rank_map.get(doc, 10**9) <= k for rank_map in ranks.values()) for doc in candidates}
        eligible = [doc for doc in candidates if doc not in set(anchor[:4])]
    else:
        raise ValueError(policy)
    if not eligible:
        return None, {"candidate_pool": len(candidates), "eligible": 0, "votes": {}, "rrf": {}}
    ordered = sorted(eligible, key=lambda doc: (-_rrf({view: rank_map[doc] for view, rank_map in ranks.items() if doc in rank_map}), -votes.get(doc, 0), min(rank_map.get(doc, 10**9) for rank_map in ranks.values()), _num(doc)))
    chosen = ordered[0]
    return chosen, {"candidate_pool": len(candidates), "eligible": len(eligible), "votes": {doc: votes.get(doc, 0) for doc in eligible}, "rrf": {doc: _rrf({view: rank_map[doc] for view, rank_map in ranks.items() if doc in rank_map}) for doc in eligible}}


def _apply_policy(questions: dict[str, dict[str, Any]], views: dict[str, dict[str, list[dict[str, Any]]]], policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for qid in sorted(questions, key=_num):
        question = questions[qid]
        selected, detail = _candidate(views[qid], list(question["anchor"]), policy)
        old = list(question["anchor"])
        old_set = set(old) & set(question["references"])
        if selected is None or selected == old[4]:
            answer = old
            swapped = False
        else:
            answer = old[:4] + [selected]
            swapped = True
        refs = set(question["references"])
        old_recall = len(set(old) & refs) / max(len(refs), 1)
        new_recall = len(set(answer) & refs) / max(len(refs), 1)
        gain = new_recall - old_recall
        rows.append({"qid": qid, "policy": policy, "answer": answer, "old_answer": old, "challenger": selected, "swapped": swapped, "is_gold": bool(selected in refs if selected else False), "old_slot5_is_gold": old[4] in refs, "recall_old": old_recall, "recall_new": new_recall, "gain": gain, "rescue": gain > 0, "destructive": gain < 0, "neutral": gain == 0, **detail})
    swaps = [row for row in rows if row["swapped"]]
    rescue = sum(row["rescue"] for row in rows)
    destructive = sum(row["destructive"] for row in rows)
    neutral = sum(row["neutral"] for row in rows if row["swapped"])
    precision = rescue / max(rescue + destructive, 1)
    summary = {"policy": policy, "queries": len(rows), "candidate_swaps": len(swaps), "rescue": rescue, "destructive": destructive, "neutral_swaps": neutral, "abstain_or_same_slot5": len(rows) - len(swaps), "precision_rescue_vs_destructive": precision, "recall_old": sum(row["recall_old"] for row in rows) / max(len(rows), 1), "recall_new": sum(row["recall_new"] for row in rows) / max(len(rows), 1), "recall_gain": sum(row["gain"] for row in rows) / max(len(rows), 1), "top1_4_identical": all(row["answer"][:4] == row["old_answer"][:4] for row in rows), "empty_output": any(not row["answer"] for row in rows), "pareto": {f"destructive_le_{limit}": max((point for point in _pareto(rows) if point["destructive"] <= limit), key=lambda point: (point["rescue"], -point["destructive"]), default=None) for limit in (0, 1, 2)}}
    return summary, rows


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Thresholding is not applied to native vote policies; this is the direct
    # D=0/1/2 audit required for the fixed policies and their actual swaps.
    points = []
    for selected in [row for row in rows if row["swapped"]]:
        chosen = [row for row in rows if row["swapped"] and row["challenger"] == selected["challenger"]]
        # Keep this compact: each policy already has one deterministic action
        # per query, so the useful point is the policy's realized outcome.
        if len(chosen) == len([row for row in rows if row["swapped"]]):
            break
    rescue = sum(row["rescue"] for row in rows)
    destructive = sum(row["destructive"] for row in rows)
    return [{"rule": "fixed_policy", "selected": sum(row["swapped"] for row in rows), "rescue": rescue, "destructive": destructive, "precision": rescue / max(rescue + destructive, 1)}]


def _agreement(questions: dict[str, dict[str, Any]], views: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    patterns: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "gold": 0})
    any_gold = two_agree = three_agree = 0
    two_gold = three_gold = 0
    records = []
    for qid in sorted(questions, key=_num):
        refs = set(questions[qid]["references"])
        top = {view: str(views[qid][view][0]["doc_id"]) for view in ("Q0", "Q1", "Q2")}
        unique = set(top.values())
        all_same = len(unique) == 1
        pairs = {"Q0=Q1": top["Q0"] == top["Q1"], "Q0=Q2": top["Q0"] == top["Q2"], "Q1=Q2": top["Q1"] == top["Q2"]}
        all_diff = len(unique) == 3
        any_is_gold = any(doc in refs for doc in top.values())
        pair_is_gold = ((pairs["Q0=Q1"] and top["Q0"] in refs) or (pairs["Q0=Q2"] and top["Q0"] in refs) or (pairs["Q1=Q2"] and top["Q1"] in refs))
        for name, active, doc in [("3 views cùng top1", all_same, top["Q0"]), ("Q0=Q1", pairs["Q0=Q1"], top["Q0"]), ("Q0=Q2", pairs["Q0=Q2"], top["Q0"]), ("Q1=Q2", pairs["Q1=Q2"], top["Q1"]), ("tất cả khác nhau", all_diff, None)]:
            if active:
                patterns[name]["count"] += 1
                patterns[name]["gold"] += int((doc in refs) if doc is not None else any_is_gold)
        any_gold += int(any_is_gold)
        two_agree += int(any(pairs.values()))
        three_agree += int(all_same)
        two_gold += int(pair_is_gold)
        three_gold += int(all_same and top["Q0"] in refs)
        records.append({"qid": qid, "top1": top, "any_view_top1_is_gold": any_is_gold, "two_views_agree_top1": any(pairs.values()), "three_views_agree_top1": all_same})
    for row in patterns.values():
        row["precision"] = row["gold"] / max(row["count"], 1)
    return {"queries": len(questions), "any_view_top1_is_gold": {"count": any_gold, "rate": any_gold / len(questions)}, "2_views_agree_top1": {"count": two_agree, "gold": two_gold, "precision": two_gold / max(two_agree, 1)}, "3_views_agree_top1": {"count": three_agree, "gold": three_gold, "precision": three_gold / max(three_agree, 1)}, "patterns": dict(patterns), "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    views = _load_views(args.views, set(questions))
    agreement = _agreement(questions, views)
    policies = ["top1_vote2", "top3_vote2", "top5_vote2", "top10_vote2", "rrf_top20"]
    summaries = []
    for policy in policies:
        summary, rows = _apply_policy(questions, views, policy)
        summaries.append(summary)
        write_jsonl(args.output_dir / f"{policy}_predictions.jsonl", rows)
    view_report = {"kind": "legalir_v33_a0_view_agreement", "schema_version": 1, "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "views_sha256": sha256_file(args.views)}, **agreement}
    write_json(args.output_dir / "view_agreement_report.json", view_report)
    write_json(args.output_dir / "consensus_pareto.json", {"kind": "legalir_v33_a1_consensus_pareto", "policies": summaries[:4]})
    write_json(args.output_dir / "rrf_report.json", {"kind": "legalir_v33_a1_rrf", "policy": summaries[4]})
    gate = {"pass": any(row["rescue"] >= 12 and row["destructive"] <= 1 and row["precision_rescue_vs_destructive"] >= 0.90 for row in summaries), "near_pass": any(8 <= row["rescue"] <= 11 and row["destructive"] <= 1 for row in summaries)}
    write_json(args.output_dir.parent / "v33_a_gate.json", {"kind": "legalir_v33_a_gate", "agreement": agreement, "policies": summaries, "gate": gate})
    print(json.dumps({"agreement": {key: value for key, value in agreement.items() if key != "records"}, "policies": summaries, "gate": gate}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
