from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import config
from .evaluate import evaluate_file
from .preprocess import preprocess_mdace
from .query_retrieval import run_query_extraction
from .recall import run_recall
from .rerank import run_rerank
from .verifier import run_verifier


def default_run_id() -> str:
    return datetime.now().strftime("mdace_cm_discharge_%Y%m%d_%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KREL MDACE ICD-10-CM discharge-summary pipeline.")
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--input-csv", default=str(config.MDACE_DIR / "mdace_icd10cm_test.csv"))
    parser.add_argument("--notes-csv", default="")
    parser.add_argument("--queries-csv", default="")
    parser.add_argument("--label-space", choices=["full", "observed"], default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-verifier", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--llm-model", default=config.LLM_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=32768)
    parser.add_argument("--query-max-concurrency", type=int, default=1)
    parser.add_argument("--verifier-max-concurrency", type=int, default=1)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--no-sectionize", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=config.RERANK_MODEL)
    parser.add_argument("--rerank-topk", type=int, default=50)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=50)
    rule_group = parser.add_mutually_exclusive_group()
    rule_group.add_argument("--use-neo4j-rules", dest="use_neo4j_rules", action="store_true", default=False)
    rule_group.add_argument("--no-neo4j-rules", dest="use_neo4j_rules", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = config.OUTPUTS_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    notes_csv = Path(args.notes_csv) if args.notes_csv else run_dir / "mdace_cm_discharge_notes.csv"
    queries_csv = Path(args.queries_csv) if args.queries_csv else run_dir / "mdace_cm_discharge_queries.csv"
    query_candidates = run_dir / "query_candidates_cm.jsonl"
    recall_results = run_dir / "recall_results_cm.csv"
    reranked_candidates = run_dir / "reranked_candidates_cm.jsonl"
    rerank_metrics = run_dir / "rerank_metrics_cm.json"
    final_predictions = run_dir / "final_predictions_cm.csv"
    evaluation_metrics = run_dir / "evaluation_metrics_cm.json"

    if not args.skip_preprocess:
        preprocess_mdace(args.input_csv, notes_csv, limit=args.limit)
    else:
        print(f"Skipping preprocess; using notes CSV: {notes_csv}")

    if not args.skip_query:
        run_query_extraction(
            notes_csv,
            queries_csv,
            model=args.llm_model,
            max_tokens=args.llm_max_tokens,
            max_concurrency=args.query_max_concurrency,
            json_mode=not args.no_json_mode,
            stream=args.stream,
            sectionize=not args.no_sectionize,
            limit=args.limit,
        )
    else:
        print(f"Skipping query extraction; using queries CSV: {queries_csv}")

    if not args.skip_recall:
        run_recall(
            queries_csv,
            query_candidates,
            recall_results,
            query_model_name=args.embedding_model,
            label_space=args.label_space,
        )
    else:
        print(f"Skipping recall; using query candidates: {query_candidates}")

    if not args.skip_rerank:
        run_rerank(
            query_candidates,
            config.ICD_CM_CORPUS_PATH,
            reranked_candidates,
            rerank_metrics,
            model_name=args.rerank_model,
            topk=args.rerank_topk,
            batch_size=args.rerank_batch_size,
        )
    else:
        print(f"Skipping rerank; using reranked candidates: {reranked_candidates}")

    if not args.skip_verifier:
        run_verifier(
            notes_csv,
            reranked_candidates,
            config.ICD_CM_CORPUS_PATH,
            final_predictions,
            model=args.llm_model,
            max_candidates=args.max_candidates,
            max_tokens=args.llm_max_tokens,
            max_concurrency=args.verifier_max_concurrency,
            json_mode=not args.no_json_mode,
            stream=args.stream,
            use_neo4j_rules=args.use_neo4j_rules,
            sectionize=not args.no_sectionize,
            limit=args.limit,
        )
    else:
        print(f"Skipping verifier; using final predictions: {final_predictions}")

    if not args.skip_evaluate:
        evaluate_file(final_predictions, evaluation_metrics)
    else:
        print(f"Skipping evaluation; final predictions: {final_predictions}")

    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
