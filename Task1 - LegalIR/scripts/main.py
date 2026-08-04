"""
Legal Information Retrieval - Main Entry Point
=============================================
Task 1 - DSC 2026 UIT Competition

Usage:
    python scripts/main.py --mode train --warmup path/to/warmup.json
    python scripts/main.py --mode infer --warmup path/to/warmup.json --output path/to/output.json
    python scripts/main.py --mode eval --predictions path/to/predictions.json --ground-truth path/to/warmup.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Legal Information Retrieval Pipeline - DSC 2026"
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'infer', 'eval', 'analyze'],
        required=True,
        help='Operation mode: train (build index), infer (run inference), eval (evaluate), analyze (warmup analysis)'
    )

    # Data paths
    parser.add_argument(
        '--warmup',
        type=str,
        help='Path to warmup.json file'
    )

    parser.add_argument(
        '--corpus',
        type=str,
        help='Path to legal corpus directory or JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for predictions'
    )

    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to predictions file for evaluation'
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        help='Directory to save/load the index'
    )

    # Model configuration
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help='Embedding model name'
    )

    parser.add_argument(
        '--reranker-model',
        type=str,
        default='cross-encoder/ms-marco-MiniLM-L-12-v2',
        help='Reranker model name'
    )

    # Retrieval configuration
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return per query'
    )

    parser.add_argument(
        '--bm25-weight',
        type=float,
        default=0.4,
        help='Weight for BM25 scores in fusion'
    )

    parser.add_argument(
        '--no-reranker',
        action='store_true',
        help='Disable cross-encoder reranking'
    )

    # Processing options
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Chunk size for document splitting'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding inference'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device for inference'
    )

    return parser


def train_mode(args):
    """Build and save the retrieval index."""
    from .pipeline import HybridRetriever
    from .corpus_loader import LegalCorpusLoader

    logger.info("="*60)
    logger.info("TRAIN MODE - Building Retrieval Index")
    logger.info("="*60)

    # Load corpus
    corpus_loader = LegalCorpusLoader()

    if args.corpus:
        documents = corpus_loader.load_corpus(Path(args.corpus))
    elif args.warmup:
        # For warmup, we need the actual corpus
        # In competition, this would be provided separately
        logger.warning("No corpus provided. Using warmup data as placeholder.")
        documents = corpus_loader.load_corpus(Path(args.warmup))
    else:
        raise ValueError("Either --corpus or --warmup must be provided")

    # Convert to list format
    doc_list = [
        {'doc_id': doc_id, 'text': content}
        for doc_id, content in documents.items()
    ]

    # Initialize retriever
    retriever = HybridRetriever(
        bm25_weight=args.bm25_weight,
        dense_weight=1.0 - args.bm25_weight,
        use_reranker=not args.no_reranker,
        use_query_expansion=True
    )

    # Build index
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/processed/index")
    retriever.index_documents(doc_list, save_path=index_dir)

    logger.info(f"Index saved to {index_dir}")
    logger.info("Training complete!")


def infer_mode(args):
    """Run inference on queries."""
    from .pipeline import HybridRetriever
    from .evaluate import LegalIREvaluator, WarmupProcessor

    logger.info("="*60)
    logger.info("INFER MODE - Running Inference")
    logger.info("="*60)

    if not args.warmup:
        raise ValueError("--warmup is required for inference mode")

    # Load warmup queries
    warmup_path = Path(args.warmup)
    queries = WarmupProcessor.load_warmup_queries(warmup_path)
    logger.info(f"Loaded {len(queries)} queries from warmup")

    # Load or build index
    index_dir = Path(args.index_dir) if args.index_dir else Path("data/processed/index")

    if index_dir.exists():
        logger.info(f"Loading existing index from {index_dir}")
        retriever = HybridRetriever.load(index_dir)
    else:
        logger.warning("No index found. Building new index...")
        from .corpus_loader import LegalCorpusLoader
        from .pipeline import HybridRetriever

        # Try to load corpus
        corpus_loader = LegalCorpusLoader()
        if args.corpus:
            documents = corpus_loader.load_corpus(Path(args.corpus))
        else:
            logger.error("No corpus provided. Cannot build index.")
            return

        doc_list = [
            {'doc_id': doc_id, 'text': content}
            for doc_id, content in documents.items()
        ]

        retriever = HybridRetriever(
            bm25_weight=args.bm25_weight,
            dense_weight=1.0 - args.bm25_weight,
            use_reranker=not args.no_reranker
        )
        retriever.index_documents(doc_list, save_path=index_dir)

    # Initialize evaluator
    evaluator = LegalIREvaluator(retriever)
    evaluator.load_ground_truth(warmup_path)

    # Run inference
    results = evaluator.run_inference(
        output_path=Path(args.output) if args.output else None,
        top_k=args.top_k
    )

    # Save submission
    output_path = Path(args.output) if args.output else Path("output/submission.json")
    submission = evaluator.save_submission(output_path, top_k=args.top_k)

    logger.info(f"Submission saved to {output_path}")
    logger.info("Inference complete!")


def eval_mode(args):
    """Evaluate predictions against ground truth."""
    from .evaluate import LegalIREvaluator, WarmupProcessor

    logger.info("="*60)
    logger.info("EVAL MODE - Evaluating Predictions")
    logger.info("="*60)

    if not args.predictions:
        raise ValueError("--predictions is required for eval mode")

    if not args.warmup:
        raise ValueError("--warmup (ground truth) is required for eval mode")

    # Load ground truth
    warmup_path = Path(args.warmup)
    gt_data = WarmupProcessor.load_warmup_queries(warmup_path)
    gt_dict = {qid: {'question': q, 'answers': set()} for qid, q in gt_data}

    # Reconstruct ground truth from warmup
    import json
    with open(warmup_path, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)

    for qid, item in warmup_data.items():
        gt_dict[qid]['answers'] = set(item.get('answer', []))

    # Load predictions
    pred_path = Path(args.predictions)
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Convert predictions format
    pred_dict = {}
    for qid, pred in predictions.items():
        if isinstance(pred, dict) and 'answer' in pred:
            pred_dict[qid] = {'answers': pred['answer']}
        elif isinstance(pred, dict) and 'answers' in pred:
            pred_dict[qid] = pred
        else:
            # Assume it's already in answer format
            pred_dict[qid] = {'answers': pred if isinstance(pred, list) else [pred]}

    # Create evaluator with empty retriever (just for metrics)
    class DummyRetriever:
        pass

    evaluator = LegalIREvaluator(DummyRetriever())
    evaluator.ground_truth = gt_dict
    evaluator.predictions = pred_dict

    # Evaluate
    metrics = evaluator.evaluate()

    # Print results
    evaluator.print_metrics(metrics)

    logger.info("Evaluation complete!")


def analyze_mode(args):
    """Analyze warmup data."""
    from .evaluate import analyze_warmup_data, print_analysis

    logger.info("="*60)
    logger.info("ANALYZE MODE - Warmup Data Analysis")
    logger.info("="*60)

    if not args.warmup:
        raise ValueError("--warmup is required for analyze mode")

    warmup_path = Path(args.warmup)
    analysis = analyze_warmup_data(warmup_path)
    print_analysis(analysis)


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'analyze':
        analyze_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
