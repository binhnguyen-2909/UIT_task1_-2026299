"""
Legal IR Inference and Evaluation
================================
Handles inference on warmup data and evaluation metrics.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

from .pipeline import HybridRetriever, QueryResult

logger = logging.getLogger(__name__)


class LegalIREvaluator:
    """
    Evaluator for Legal Information Retrieval.

    Metrics:
    - MRR (Mean Reciprocal Rank)
    - Recall@K
    - Precision@K
    - NDCG@K
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.ground_truth = {}
        self.predictions = {}

    def load_ground_truth(self, path: Path):
        """Load ground truth from warmup.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.ground_truth = {}
        for query_id, item in data.items():
            self.ground_truth[query_id] = {
                'question': item['question'],
                'answers': set(item.get('answer', []))
            }

        logger.info(f"Loaded {len(self.ground_truth)} ground truth queries")
        return self.ground_truth

    def run_inference(
        self,
        output_path: Optional[Path] = None,
        top_k: int = 5,
        save_results: bool = True
    ) -> List[QueryResult]:
        """
        Run inference on all ground truth queries.

        Args:
            output_path: Path to save predictions
            top_k: Number of results to return per query
            save_results: Whether to save predictions

        Returns:
            List of QueryResult objects
        """
        results = []
        predictions = {}

        logger.info(f"Running inference on {len(self.ground_truth)} queries...")

        total_time = 0
        for query_id, gt_item in self.ground_truth.items():
            query = gt_item['question']

            result = self.retriever.retrieve(query, query_id, top_k=top_k)
            results.append(result)

            # Store prediction (just doc_ids)
            predictions[query_id] = {
                'question': query,
                'answers': [r.doc_id for r in result.results],
                'processing_time': result.processing_time
            }

            total_time += result.processing_time

            if len(results) % 50 == 0:
                logger.info(f"Processed {len(results)}/{len(self.ground_truth)} queries")

        avg_time = total_time / len(results) if results else 0
        logger.info(f"Inference complete. Avg time per query: {avg_time*1000:.1f}ms")

        self.predictions = predictions

        if save_results and output_path:
            self.save_predictions(output_path)

        return results

    def evaluate(
        self,
        predictions: Optional[Dict] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Optional dict of predictions (uses self.predictions if None)
            k_values: K values for Recall@K and Precision@K

        Returns:
            Dictionary of evaluation metrics
        """
        if predictions is None:
            predictions = self.predictions

        metrics = {}

        # Initialize accumulators
        mrr_sum = 0
        recall_sums = {k: 0 for k in k_values}
        precision_sums = {k: 0 for k in k_values}
        ndcg_sums = {k: 0 for k in k_values}

        num_queries = 0

        for query_id, gt in self.ground_truth.items():
            if query_id not in predictions:
                logger.warning(f"No prediction for query {query_id}")
                continue

            pred = predictions[query_id]
            pred_answers = pred.get('answers', [])
            gt_answers = gt['answers']

            num_queries += 1

            # MRR
            mrr = self._compute_mrr(pred_answers, gt_answers)
            mrr_sum += mrr

            # Recall@K
            for k in k_values:
                recall_sums[k] += self._compute_recall(pred_answers[:k], gt_answers)

            # Precision@K
            for k in k_values:
                precision_sums[k] += self._compute_precision(pred_answers[:k], gt_answers)

            # NDCG@K
            for k in k_values:
                ndcg_sums[k] += self._compute_ndcg(pred_answers[:k], gt_answers)

        # Average
        if num_queries > 0:
            metrics['MRR'] = mrr_sum / num_queries

            for k in k_values:
                metrics[f'Recall@{k}'] = recall_sums[k] / num_queries
                metrics[f'Precision@{k}'] = precision_sums[k] / num_queries
                metrics[f'NDCG@{k}'] = ndcg_sums[k] / num_queries

        metrics['num_queries'] = num_queries
        metrics['num_ground_truth_docs'] = sum(len(gt['answers']) for gt in self.ground_truth.values())
        metrics['avg_pred_per_query'] = sum(len(p.get('answers', [])) for p in predictions.values()) / len(predictions) if predictions else 0

        return metrics

    def _compute_mrr(self, predictions: List[str], ground_truth: set) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, pred in enumerate(predictions):
            if pred in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_recall(self, predictions: List[str], ground_truth: set) -> float:
        """Compute Recall@K."""
        if not ground_truth:
            return 0.0
        hits = sum(1 for p in predictions if p in ground_truth)
        return hits / len(ground_truth)

    def _compute_precision(self, predictions: List[str], ground_truth: set) -> float:
        """Compute Precision@K."""
        if not predictions:
            return 0.0
        hits = sum(1 for p in predictions if p in ground_truth)
        return hits / len(predictions)

    def _compute_ndcg(self, predictions: List[str], ground_truth: set) -> float:
        """Compute NDCG@K."""
        # DCG
        dcg = 0.0
        for i, pred in enumerate(predictions):
            if pred in ground_truth:
                dcg += 1.0 / (i + 1)

        # IDCG (ideal DCG)
        ideal_k = min(len(ground_truth), len(predictions))
        idcg = sum(1.0 / (i + 1) for i in range(ideal_k))

        return dcg / idcg if idcg > 0 else 0.0

    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a nice format."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        print(f"\nOverall:")
        print(f"  Queries: {metrics['num_queries']}")
        print(f"  Ground Truth Docs: {metrics['num_ground_truth_docs']}")
        print(f"  Avg Predictions per Query: {metrics['avg_pred_per_query']:.1f}")

        print(f"\nMain Metrics:")
        print(f"  MRR:        {metrics.get('MRR', 0):.4f}")

        print(f"\nRecall@{list(metrics.keys())[5:9]}:")
        for k in [1, 3, 5, 10]:
            if f'Recall@{k}' in metrics:
                print(f"  Recall@{k}:   {metrics[f'Recall@{k}']:.4f}")

        print(f"\nPrecision@{list(metrics.keys())[5:9]}:")
        for k in [1, 3, 5, 10]:
            if f'Precision@{k}' in metrics:
                print(f"  Precision@{k}: {metrics[f'Precision@{k}']:.4f}")

        print(f"\nNDCG@{list(metrics.keys())[5:9]}:")
        for k in [1, 3, 5, 10]:
            if f'NDCG@{k}' in metrics:
                print(f"  NDCG@{k}:     {metrics[f'NDCG@{k}']:.4f}")

        print("\n" + "="*60)

    def save_predictions(self, path: Path):
        """Save predictions to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=2)

        logger.info(f"Predictions saved to {path}")

    def save_submission(self, path: Path, top_k: int = 5):
        """
        Save predictions in submission format.

        Format: {query_id: {answers: [doc_id1, doc_id2, ...]}}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        submission = {}
        for query_id, pred in self.predictions.items():
            submission[query_id] = {
                'answer': pred.get('answers', [])[:top_k]
            }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)

        logger.info(f"Submission saved to {path}")
        return submission


class WarmupProcessor:
    """
    Process warmup.json to extract corpus and queries.
    """

    @staticmethod
    def extract_corpus_from_warmup(warmup_path: Path) -> Dict[str, str]:
        """
        Extract unique document IDs from warmup.json answers.

        Note: This only gives us the documents that appear in answers,
        not the full corpus. A full solution would need the actual corpus.
        """
        with open(warmup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract all document IDs from answers
        doc_ids = set()
        for item in data.values():
            if 'answer' in item:
                doc_ids.update(item['answer'])

        logger.info(f"Found {len(doc_ids)} unique document IDs in warmup")

        return {doc_id: doc_id for doc_id in doc_ids}  # Placeholder

    @staticmethod
    def load_warmup_queries(warmup_path: Path) -> List[Tuple[str, str]]:
        """Load queries from warmup.json."""
        with open(warmup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = []
        for query_id, item in data.items():
            queries.append((query_id, item['question']))

        logger.info(f"Loaded {len(queries)} queries from warmup")
        return queries


# Analysis of warmup data
def analyze_warmup_data(warmup_path: Path) -> Dict:
    """Analyze warmup.json to understand the data distribution."""
    with open(warmup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Stats
    num_queries = len(data)
    num_answers = []
    answer_counts = Counter()

    for item in data.values():
        answers = item.get('answer', [])
        num_answers.append(len(answers))
        answer_counts[len(answers)] += 1

        # Track individual doc frequencies
        for doc_id in answers:
            answer_counts[doc_id] += 1

    analysis = {
        'num_queries': num_queries,
        'total_ground_truth_docs': sum(num_answers),
        'avg_answers_per_query': sum(num_answers) / num_queries if num_queries else 0,
        'max_answers': max(num_answers) if num_answers else 0,
        'min_answers': min(num_answers) if num_answers else 0,
        'answer_distribution': dict(answer_counts),
        'num_unique_ground_truth_docs': len(set(doc for item in data.values() for doc in item.get('answer', []))),
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print warmup data analysis."""
    print("\n" + "="*60)
    print("WARMUP DATA ANALYSIS")
    print("="*60)

    print(f"\nDataset Statistics:")
    print(f"  Total queries: {analysis['num_queries']}")
    print(f"  Total ground truth documents: {analysis['total_ground_truth_docs']}")
    print(f"  Unique ground truth documents: {analysis['num_unique_ground_truth_docs']}")
    print(f"  Avg answers per query: {analysis['avg_answers_per_query']:.2f}")
    print(f"  Max answers for a query: {analysis['max_answers']}")
    print(f"  Min answers for a query: {analysis['min_answers']}")

    print(f"\nAnswer Count Distribution:")
    for count, freq in sorted(analysis['answer_distribution'].items()):
        if count <= 10:  # Only show small counts
            print(f"  {count} answers: {freq} queries ({100*freq/analysis['num_queries']:.1f}%)")

    print("\n" + "="*60)


# Main execution
if __name__ == "__main__":
    # Example analysis
    warmup_path = Path("d:/projects/UIT DSC/Task1 - LegalIR/warmup.json")

    if warmup_path.exists():
        analysis = analyze_warmup_data(warmup_path)
        print_analysis(analysis)
    else:
        print(f"Warmup file not found: {warmup_path}")
