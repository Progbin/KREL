from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .io_utils import compute_set_metrics, normalize_code, parse_ground_truth, parse_prediction_codes


def parse_candidate_codes(raw: Any) -> set[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    if isinstance(raw, list):
        return {normalize_code(code) for code in raw if str(code).strip()}
    text = str(raw).strip()
    if not text:
        return set()
    return {normalize_code(code) for code in text.split(";") if code.strip()}


def has_error_value(raw: Any) -> bool:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return False
    return bool(str(raw).strip())


def evaluate_file(
    predictions_csv: str | Path,
    output_json: str | Path,
    *,
    positive_verdicts: set[str] | None = None,
) -> dict[str, Any]:
    positive_verdicts = positive_verdicts or {"SUPPORTED", "POSSIBLE"}
    df = pd.read_csv(predictions_csv)
    golds: list[set[str]] = []
    preds: list[set[str]] = []
    candidates: list[set[str]] = []
    errors = 0

    for _, row in df.iterrows():
        gt = parse_ground_truth(row.get("ground_truth"))
        pred = parse_prediction_codes(row.get("final_predict"), positive=positive_verdicts)
        cand = parse_candidate_codes(row.get("candidate_codes"))
        if has_error_value(row.get("error")):
            errors += 1
        golds.append(gt)
        preds.append(pred)
        candidates.append(cand)

    pred_metrics = compute_set_metrics(golds, preds)
    candidate_metrics = compute_set_metrics(golds, candidates)
    candidate_sizes = [len(item) for item in candidates]
    pred_sizes = [len(item) for item in preds]

    metrics: dict[str, Any] = {
        "positive_verdicts": sorted(positive_verdicts),
        "prediction": pred_metrics,
        "candidate_input": candidate_metrics,
        "num_rows_with_errors": errors,
        "avg_prediction_size": sum(pred_sizes) / len(pred_sizes) if pred_sizes else 0.0,
        "avg_candidate_set_size": sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KREL ICD-10-CM predictions.")
    parser.add_argument("--predictions-csv", default=str(config.OUTPUTS_DIR / "final_predictions_cm.csv"))
    parser.add_argument("--output-json", default=str(config.OUTPUTS_DIR / "evaluation_metrics_cm.json"))
    parser.add_argument(
        "--positive-verdicts",
        default="SUPPORTED,POSSIBLE",
        help="Comma-separated verdicts counted as predicted positives.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    positive = {item.strip().upper() for item in args.positive_verdicts.split(",") if item.strip()}
    evaluate_file(args.predictions_csv, args.output_json, positive_verdicts=positive)


if __name__ == "__main__":
    main()
