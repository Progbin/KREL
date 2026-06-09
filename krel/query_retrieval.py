from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import config
from .io_utils import extract_queries
from .llm import chat_json, dumps_json_response
from .prompts import QUERY_SYSTEM_PROMPT_CM, make_query_user_prompt
from .sectionizer import sectionize_text


def _completed_previous(output_csv: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(output_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "note_id" not in df.columns or "respond" not in df.columns:
        return {}
    previous: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        respond = row.get("respond")
        error = str(row.get("error") or "").strip()
        if isinstance(respond, str) and respond.strip() and not error:
            previous[str(row["note_id"])] = row.to_dict()
    return previous


def _extract_one(
    row: dict[str, Any],
    *,
    model: str | None,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    sectionize: bool,
) -> dict[str, Any]:
    out = dict(row)
    try:
        note_text = str(row.get("text") or "")
        if sectionize:
            note_text = sectionize_text(note_text)
        payload, raw = chat_json(
            QUERY_SYSTEM_PROMPT_CM,
            make_query_user_prompt(note_text),
            model=model,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=stream,
        )
        out["respond"] = dumps_json_response(payload)
        out["query_count"] = len(extract_queries(raw))
        out["error"] = ""
    except Exception as exc:
        out["respond"] = ""
        out["query_count"] = 0
        out["error"] = repr(exc)
    return out


def run_query_extraction(
    notes_csv: str | Path,
    output_csv: str | Path,
    *,
    model: str | None = None,
    max_tokens: int = 32768,
    max_concurrency: int = 1,
    json_mode: bool = True,
    stream: bool = False,
    sectionize: bool = True,
    limit: int | None = None,
    resume: bool = True,
    save_every: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(notes_csv)
    if limit is not None:
        df = df.head(limit).copy()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    previous = _completed_previous(output_csv) if resume else {}

    rows = df.to_dict("records")
    outputs: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        old = previous.get(str(row.get("note_id")))
        if old is not None:
            merged = dict(row)
            merged.update({k: old.get(k) for k in ("respond", "query_count", "error") if k in old})
            outputs[idx] = merged
        else:
            pending.append((idx, row))

    def flush() -> None:
        materialized = [item for item in outputs if item is not None]
        pd.DataFrame(materialized).to_csv(output_csv, index=False)

    if max_concurrency <= 1:
        for done_count, (idx, row) in enumerate(tqdm(pending, desc="query extraction"), start=1):
            outputs[idx] = _extract_one(
                row,
                model=model,
                max_tokens=max_tokens,
                json_mode=json_mode,
                stream=stream,
                sectionize=sectionize,
            )
            if done_count % save_every == 0:
                flush()
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(
                    _extract_one,
                    row,
                    model=model,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    stream=stream,
                    sectionize=sectionize,
                ): idx
                for idx, row in pending
            }
            done_count = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="query extraction"):
                idx = futures[future]
                outputs[idx] = future.result()
                done_count += 1
                if done_count % save_every == 0:
                    flush()

    flush()
    return pd.DataFrame([item for item in outputs if item is not None])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ICD-oriented query extraction for MDACE notes.")
    parser.add_argument("--notes-csv", default=str(config.OUTPUTS_DIR / "mdace_cm_discharge_notes.csv"))
    parser.add_argument("--output-csv", default=str(config.OUTPUTS_DIR / "mdace_cm_discharge_queries.csv"))
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--no-sectionize", action="store_true")
    parser.add_argument("--stream", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = run_query_extraction(
        args.notes_csv,
        args.output_csv,
        model=args.model,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        json_mode=not args.no_json_mode,
        sectionize=not args.no_sectionize,
        stream=args.stream,
        limit=args.limit,
        resume=not args.no_resume,
    )
    ok = int((df.get("error", "").astype(str).str.len() == 0).sum()) if "error" in df.columns else len(df)
    print(f"Wrote {len(df)} rows to {args.output_csv}; successful rows: {ok}")


if __name__ == "__main__":
    main()
