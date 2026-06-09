from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def normalize_code(code: Any) -> str:
    return str(code).strip().replace(".", "").upper()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    for start in sorted(starts):
        try:
            _, end = decoder.raw_decode(text[start:])
            return text[start : start + end]
        except Exception:
            continue
    return text


def parse_jsonish(text: Any) -> Any:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    raw = extract_json_text(str(text))
    if not raw:
        return None
    return json.loads(raw)


def _query_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value or None
    if not isinstance(item, dict):
        return None
    base = str(item.get("base") or "").strip()
    query = str(item.get("query") or "").strip()
    if query:
        return query
    if base:
        return base
    for key in ("diagnosis", "condition", "problem", "name", "text"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _diagnosis_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        return []
    items: list[Any] = []
    pr = payload.get("principal_reason")
    if pr:
        items.append(pr)
    for key in ("active_conditions", "history_conditions", "queries", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(value)
    return items


def extract_query_items(raw_response: Any) -> list[dict[str, Any]]:
    try:
        payload = parse_jsonish(raw_response)
    except Exception:
        return []
    items = _diagnosis_items(payload)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        query = _query_from_item(item)
        if not query:
            continue
        key = " ".join(query.lower().split())
        if key in seen:
            continue
        seen.add(key)
        if isinstance(item, dict):
            record = dict(item)
        else:
            record = {"base": query}
        record["query"] = query
        output.append(record)
    return output


def extract_queries(raw_response: Any) -> list[str]:
    queries: list[str] = []
    for item in extract_query_items(raw_response):
        query = str(item.get("query") or "").strip()
        if query:
            queries.append(query)
    return queries


def parse_ground_truth(raw: Any) -> set[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    return {normalize_code(code) for code in str(raw).split(";") if str(code).strip()}


def parse_prediction_codes(raw: Any, positive: set[str] | None = None) -> set[str]:
    positive = positive or {"SUPPORTED", "POSSIBLE"}
    try:
        payload = parse_jsonish(raw)
    except Exception:
        return set()
    if not isinstance(payload, dict):
        return set()
    out: set[str] = set()
    for item in payload.get("verifications") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().upper()
        code = item.get("code")
        if code and verdict in positive:
            out.add(normalize_code(code))
    return out


def load_corpus(corpus_path: str | Path) -> dict[str, dict[str, Any]]:
    code2entry: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(corpus_path):
        code = None
        if isinstance(row.get("metadata"), dict):
            code = row["metadata"].get("code")
        code = code or row.get("code")
        if code:
            code2entry[normalize_code(code)] = row
    return code2entry


def code_description(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    if isinstance(entry.get("page_content"), str):
        return " ".join(entry["page_content"].split())
    for key in ("description", "definition", "title", "desc"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def compute_set_metrics(golds: list[set[str]], preds: list[set[str]]) -> dict[str, float | int]:
    tp = fp = fn = 0
    macro_p = []
    macro_r = []
    macro_f = []
    for gt, pred in zip(golds, preds):
        t = len(gt & pred)
        fpos = len(pred - gt)
        fneg = len(gt - pred)
        p = t / (t + fpos) if t + fpos else 0.0
        r = t / (t + fneg) if t + fneg else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        tp += t
        fp += fpos
        fn += fneg
        macro_p.append(p)
        macro_r.append(r)
        macro_f.append(f1)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = len(golds)
    return {
        "units": n,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "micro_precision": p,
        "micro_recall": r,
        "micro_f1": f1,
        "macro_precision": sum(macro_p) / n if n else 0.0,
        "macro_recall": sum(macro_r) / n if n else 0.0,
        "macro_f1": sum(macro_f) / n if n else 0.0,
    }
