from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config


REQUIRED_COLUMNS = ["hadm_id", "note_id", "note_type", "note_subtype", "text", "ground_truth"]


def preprocess_mdace(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    note_type: str = "Discharge summary",
    discharge_only: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    if discharge_only:
        df = df[df["note_type"].astype(str).str.lower() == note_type.lower()].copy()
    if limit is not None:
        df = df.head(limit).copy()

    df = df[REQUIRED_COLUMNS].copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MDACE notes for the KREL ICD-10-CM pipeline.")
    parser.add_argument("--input", default=str(config.MDACE_DIR / "mdace_icd10cm_test.csv"))
    parser.add_argument("--output", default=str(config.OUTPUTS_DIR / "mdace_cm_discharge_notes.csv"))
    parser.add_argument("--note-type", default="Discharge summary")
    parser.add_argument("--all-note-types", action="store_true", help="Do not filter to discharge summaries.")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = preprocess_mdace(
        args.input,
        args.output,
        note_type=args.note_type,
        discharge_only=not args.all_note_types,
        limit=args.limit,
    )
    print(f"Wrote {len(df)} notes to {args.output}")


if __name__ == "__main__":
    main()
