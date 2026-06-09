from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import config
from .io_utils import code_description, load_corpus, normalize_code, read_jsonl
from .llm import chat_json, dumps_json_response
from .prompts import VERIFIER_SYSTEM_PROMPT_CM, make_verifier_user_prompt
from .sectionizer import sectionize_text


MAX_CANDIDATES = 50


def group_rows_by_note(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    note2rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        note_id = str(row.get("note_id") or "")
        if note_id:
            note2rows[note_id].append(row)
    return dict(note2rows)


def pair_builder(row: dict[str, Any]) -> list[tuple[str, float]]:
    codes = row.get("reranked_codes") or row.get("candidates") or []
    scores = row.get("reranked_scores") or row.get("candidate_scores") or []
    pairs: list[tuple[str, float]] = []
    for idx, code in enumerate(codes):
        try:
            score = float(scores[idx])
        except Exception:
            score = 0.0
        pairs.append((str(code), score))
    return pairs


def select_candidates_for_note_dynamic(
    query_rows: list[dict[str, Any]],
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[list[str], dict[str, float], dict[int, list[str]]]:
    if not query_rows:
        return [], {}, {}

    query_count = len(query_rows)
    base = max_candidates // query_count
    remainder = max_candidates - base * query_count

    per_query_selected: dict[int, list[str]] = {}
    code2score: dict[str, float] = {}
    norm2raw: dict[str, str] = {}
    selected_norms: set[str] = set()
    leftovers: list[tuple[str, float]] = []

    for idx, row in enumerate(query_rows):
        pairs = pair_builder(row)
        quota = base + (1 if idx < remainder else 0)
        chosen = pairs[:quota]
        per_query_selected[idx] = [code for code, _ in chosen]
        for code, score in chosen:
            normalized = normalize_code(code)
            selected_norms.add(normalized)
            code2score[normalized] = max(code2score.get(normalized, float("-inf")), float(score))
            norm2raw[normalized] = code
        leftovers.extend((code, float(score)) for code, score in pairs[quota:])

    if len(selected_norms) < max_candidates and leftovers:
        leftovers.sort(key=lambda item: item[1], reverse=True)
        for code, score in leftovers:
            normalized = normalize_code(code)
            if normalized in selected_norms:
                continue
            selected_norms.add(normalized)
            code2score[normalized] = max(code2score.get(normalized, float("-inf")), score)
            norm2raw[normalized] = code
            if len(selected_norms) >= max_candidates:
                break

    ranked_norms = sorted(code2score.keys(), key=lambda code: code2score[code], reverse=True)[:max_candidates]
    return [norm2raw[code] for code in ranked_norms], code2score, per_query_selected


def _format_evidence(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        lines: list[str] = []
        for item in value[:5]:
            if isinstance(item, str) and item.strip():
                lines.append(item.strip())
            elif isinstance(item, dict):
                section = item.get("section") or item.get("sec") or item.get("header")
                span = str(item.get("span") or item.get("text") or "").strip()
                if span:
                    lines.append(f"[{section}] {span}" if section else span)
        return lines
    if isinstance(value, dict):
        section = value.get("section") or value.get("sec") or value.get("header")
        span = str(value.get("span") or value.get("text") or "").strip()
        if span:
            return [f"[{section}] {span}" if section else span]
    return [str(value).strip()]


def build_evidence_candidate_blocks(
    query_rows: list[dict[str, Any]],
    code2entry: dict[str, dict[str, Any]],
    *,
    per_query_selected: dict[int, list[str]] | None = None,
    rule_hints_by_code: dict[str, list[str]] | None = None,
    max_desc_chars: int = 800,
    per_query_topn: int = 10,
) -> str:
    blocks: list[str] = []
    for idx, row in enumerate(query_rows):
        query = str(row.get("query") or "").strip()
        pairs = pair_builder(row)
        if per_query_selected is not None and idx in per_query_selected:
            selected = set(per_query_selected[idx])
            pairs = [(code, score) for code, score in pairs if code in selected]
        pairs = pairs[:per_query_topn]

        lines = [f"[EVIDENCE BLOCK {idx + 1}]"]
        if query:
            lines.append(f"query: {query}")
        evidence_lines = _format_evidence(row.get("evidence") or row.get("evidence_spans") or row.get("evidence_span"))
        if evidence_lines:
            lines.append("evidence hints:")
            lines.extend(f"- {line}" for line in evidence_lines[:3])
        else:
            lines.append("evidence hints: (none)")
        lines.append("candidate codes:")
        for rank, (code, score) in enumerate(pairs, start=1):
            desc = code_description(code2entry.get(normalize_code(code)))
            if desc and len(desc) > max_desc_chars:
                desc = desc[:max_desc_chars] + "..."
            if not desc:
                desc = "(missing in corpus)"
            lines.append(f"{rank}. {code} | rerank_score={score:.6f}")
            lines.append(f"   desc: {desc}")
            if rule_hints_by_code and normalize_code(code) in rule_hints_by_code:
                for hint in rule_hints_by_code[normalize_code(code)][:5]:
                    lines.append(f"   rule: {hint}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


RULE_REL_TYPES = ("useAdditionalCode", "codeFirst", "codeAlso")
PRIMARY_RULE_REL_TYPES = frozenset({"useAdditionalCode", "codeFirst"})
RULE_REL_ORDER = {"useAdditionalCode": 0, "codeFirst": 1, "codeAlso": 2}


def _code_lookup_keys(code: str) -> list[str]:
    raw = str(code or "").strip()
    normalized = normalize_code(raw)
    keys = [raw, normalized]
    if normalized and "." not in normalized and len(normalized) > 3:
        keys.append(f"{normalized[:3]}.{normalized[3:]}")
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def fetch_neo4j_rule_hints(
    codes: list[str],
    *,
    code2score: dict[str, float] | None = None,
    code2entry: dict[str, dict[str, Any]] | None = None,
    max_hints_per_code: int = 5,
    max_desc_chars: int = 800,
) -> dict[str, list[str]]:
    if not config.NEO4J_URI:
        return {}
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        print(f"Neo4j package unavailable; skipping rule hints: {exc}")
        return {}

    code2score = code2score or {}
    code2entry = code2entry or {}
    candidate_norms = {normalize_code(code) for code in codes if str(code).strip()}
    unique_codes: list[str] = []
    seen_sources: set[str] = set()
    for code in codes:
        normalized = normalize_code(code)
        if normalized and normalized not in seen_sources:
            seen_sources.add(normalized)
            unique_codes.append(str(code))

    ranked_hints: dict[str, list[tuple[tuple[int, float, int, str], str]]] = defaultdict(list)
    seen_rules: dict[str, set[tuple[str, str]]] = defaultdict(set)
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DB) as session:
            for code in unique_codes:
                source_norm = normalize_code(code)
                result = session.run(
                    """
                    MATCH (c)-[r]->(t)
                    WHERE (c.code IN $code_keys OR c.icd_code IN $code_keys)
                      AND type(r) IN $rel_types
                    RETURN type(r) AS rel, coalesce(t.code, t.icd_code, t.name) AS target
                    """,
                    code_keys=_code_lookup_keys(code),
                    rel_types=list(RULE_REL_TYPES),
                )
                for record in result:
                    target = str(record.get("target") or "").strip()
                    rel = str(record.get("rel") or "").strip()
                    if not target or rel not in RULE_REL_TYPES:
                        continue
                    target_norm = normalize_code(target)
                    if target_norm not in candidate_norms:
                        continue
                    dedupe_key = (rel, target_norm or target)
                    if dedupe_key in seen_rules[source_norm]:
                        continue
                    seen_rules[source_norm].add(dedupe_key)

                    relation_group = 0 if rel in PRIMARY_RULE_REL_TYPES else 1
                    score = float(code2score.get(target_norm, float("-inf")))
                    sort_key = (relation_group, -score, RULE_REL_ORDER.get(rel, 99), target_norm or target)
                    desc = code_description(code2entry.get(target_norm))
                    if desc and len(desc) > max_desc_chars:
                        desc = desc[:max_desc_chars] + "..."
                    desc_text = f" | desc: {desc}" if desc else ""
                    score_text = "" if score == float("-inf") else f" | target_rerank_score={score:.6f}"
                    ranked_hints[source_norm].append(
                        (sort_key, f"Rule: {rel} -> {target} [target_in_global_candidate_set]{score_text}{desc_text}")
                    )
        driver.close()
    except Exception as exc:
        print(f"Neo4j rule hints failed; continuing without them: {exc}")
        return {}

    hints: dict[str, list[str]] = {}
    for source_norm, items in ranked_hints.items():
        items.sort(key=lambda item: item[0])
        hints[source_norm] = [hint for _, hint in items[:max_hints_per_code]]
    return hints


def fetch_neo4j_code_ancestor_map(codes: list[str]) -> dict[str, set[str]]:
    if not config.NEO4J_URI:
        return {}
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        print(f"Neo4j package unavailable; skipping ancestor map: {exc}")
        return {}

    normalized_codes = sorted({normalize_code(code) for code in codes if str(code).strip()})
    if not normalized_codes:
        return {}
    query = """
    UNWIND $codes AS code
    MATCH (cand:ICD_Code)
    WHERE replace(toUpper(cand.code), '.', '') = code
    OPTIONAL MATCH (cand)-[:IS_A*0..]->(ancestor:ICD_Code)
    RETURN code AS code, collect(DISTINCT ancestor.code) AS ancestors
    """
    output: dict[str, set[str]] = {}
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DB) as session:
            for record in session.run(query, codes=normalized_codes):
                code = normalize_code(record.get("code"))
                ancestors = {normalize_code(value) for value in (record.get("ancestors") or []) if value}
                ancestors.add(code)
                output[code] = ancestors
        driver.close()
    except Exception as exc:
        print(f"Neo4j ancestor map failed; continuing without combination checks: {exc}")
        return {}
    return output


def fetch_neo4j_combination_rules() -> list[dict[str, Any]]:
    if not config.NEO4J_URI:
        return []
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        print(f"Neo4j package unavailable; skipping combination rules: {exc}")
        return []

    query = """
    MATCH (r:Rule {type:'combination'})-[hc:HAS_COMPONENT]->(component:ICD_Code)
    MATCH (r)-[:YIELDS]->(target:ICD_Code)
    RETURN r.id AS rule_id,
           r.logic AS logic,
           r.source AS source,
           r.allow_descendant_match AS allow_descendant_match,
           r.note AS note,
           target.code AS target_code,
           collect({
             code: component.code,
             order: hc.order,
             match_mode: hc.match_mode
           }) AS components
    ORDER BY rule_id
    """
    rules: list[dict[str, Any]] = []
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DB) as session:
            for record in session.run(query):
                row = dict(record)
                row["target_display"] = str(row.get("target_code") or "")
                row["target_code"] = normalize_code(row.get("target_code"))
                components = row.get("components") or []
                for component in components:
                    component["display"] = str(component.get("code") or "")
                    component["code"] = normalize_code(component.get("code"))
                components.sort(key=lambda item: int(item.get("order") or 0))
                rules.append(row)
        driver.close()
    except Exception as exc:
        print(f"Neo4j combination rules failed; continuing without combination checks: {exc}")
        return []
    return rules


def _format_component_label(component_code: str, match_mode: str | None) -> str:
    if match_mode == "self_or_descendant":
        return f"{component_code}*"
    return component_code


def build_combination_hints_for_note(
    note_candidate_codes: list[str],
    code2entry: dict[str, dict[str, Any]],
    *,
    max_desc_chars: int = 800,
) -> dict[str, list[str]]:
    if not note_candidate_codes:
        return {}

    note_candidate_set = {normalize_code(code) for code in note_candidate_codes if str(code).strip()}
    ancestor_map = fetch_neo4j_code_ancestor_map(note_candidate_codes)
    rules = fetch_neo4j_combination_rules()
    if not ancestor_map or not rules:
        return {}

    hints_by_code: dict[str, list[str]] = defaultdict(list)
    for rule in rules:
        components = rule.get("components") or []
        if not components:
            continue
        matched_components: list[tuple[str, list[str]]] = []
        for component in components:
            component_code = normalize_code(component.get("code"))
            match_mode = component.get("match_mode")
            if match_mode == "exact":
                component_matches = sorted(code for code in note_candidate_set if code == component_code)
            else:
                component_matches = sorted(
                    code for code in note_candidate_set if component_code in ancestor_map.get(code, {code})
                )
            if component_matches:
                matched_components.append((component_code, component_matches))
        if len(matched_components) < len(components):
            continue

        target_code = normalize_code(rule.get("target_code"))
        target_display = str(rule.get("target_display") or rule.get("target_code") or target_code)
        target_in_pool = target_code in note_candidate_set
        target_desc = code_description(code2entry.get(target_code))
        if target_desc and len(target_desc) > max_desc_chars:
            target_desc = target_desc[:max_desc_chars] + "..."

        component_specs: list[str] = []
        component_matches_text: list[str] = []
        for component in components:
            component_code = normalize_code(component.get("code"))
            component_display = str(component.get("display") or component.get("code") or component_code)
            label = _format_component_label(component_display, component.get("match_mode"))
            component_specs.append(label)
            matches = next((items for code, items in matched_components if code == component_code), [])
            if matches:
                component_matches_text.append(f"{label} via {', '.join(matches[:3])}")

        hint = (
            f"Combination rule: target {target_display} if components {' + '.join(component_specs)} "
            "are jointly supported across the note."
        )
        if target_desc:
            hint += f" target desc: {target_desc}"
        if target_in_pool:
            hint += " target is present in the global candidate set."
        else:
            hint += (
                " target is NOT present in the global candidate set, but it may be added only as a Stage 2 "
                "combination exception if all required component families are supported by the note."
            )
        if component_matches_text:
            hint += f" matched components in candidate pool: {'; '.join(component_matches_text)}"
        note = str(rule.get("note") or "").strip()
        if note:
            hint += f" rule note: {note}"
        hints_by_code[target_code or target_display].append(hint)
    return hints_by_code


def build_combination_section(combination_hints_by_code: dict[str, list[str]] | None) -> str:
    if not combination_hints_by_code:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for hints in combination_hints_by_code.values():
        for hint in hints:
            if hint in seen:
                continue
            seen.add(hint)
            lines.append(f"- {hint}")
    return "\n".join(lines)


def _verify_one(
    note_row: dict[str, Any],
    query_rows: list[dict[str, Any]],
    *,
    code2entry: dict[str, dict[str, Any]],
    model: str | None,
    max_candidates: int,
    max_tokens: int,
    json_mode: bool,
    stream: bool,
    use_neo4j_rules: bool,
    sectionize: bool,
) -> dict[str, Any]:
    note_id = str(note_row.get("note_id") or "")
    hadm_id = str(note_row.get("hadm_id") or "")
    out: dict[str, Any] = {
        "hadm_id": hadm_id,
        "note_id": note_id,
        "ground_truth": note_row.get("ground_truth", ""),
    }
    candidate_codes, code2score, per_query_selected = select_candidates_for_note_dynamic(
        query_rows,
        max_candidates=max_candidates,
    )
    out["candidate_codes"] = ";".join(candidate_codes)
    out["selected_candidate_count"] = len(candidate_codes)
    if not candidate_codes:
        out["final_predict"] = '{"verifications":[]}'
        out["error"] = ""
        return out

    rule_hints = (
        fetch_neo4j_rule_hints(candidate_codes, code2score=code2score, code2entry=code2entry)
        if use_neo4j_rules
        else {}
    )
    combination_section = (
        build_combination_section(build_combination_hints_for_note(candidate_codes, code2entry))
        if use_neo4j_rules
        else ""
    )
    evidence_blocks = build_evidence_candidate_blocks(
        query_rows,
        code2entry,
        per_query_selected=per_query_selected,
        rule_hints_by_code=rule_hints,
    )
    note_text = str(note_row.get("text") or "")
    if sectionize:
        note_text = sectionize_text(note_text)
    try:
        payload, _ = chat_json(
            VERIFIER_SYSTEM_PROMPT_CM,
            make_verifier_user_prompt(
                note_text=note_text,
                evidence_blocks=evidence_blocks,
                candidate_codes=candidate_codes,
                combination_section=combination_section,
            ),
            model=model,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=stream,
        )
        out["final_predict"] = dumps_json_response(payload)
        out["error"] = ""
    except Exception as exc:
        out["final_predict"] = ""
        out["error"] = repr(exc)
    return out


def run_verifier(
    notes_csv: str | Path,
    reranked_jsonl: str | Path,
    corpus_path: str | Path,
    output_csv: str | Path,
    *,
    model: str | None = None,
    max_candidates: int = MAX_CANDIDATES,
    max_tokens: int = 32768,
    max_concurrency: int = 1,
    json_mode: bool = True,
    stream: bool = False,
    use_neo4j_rules: bool = False,
    sectionize: bool = True,
    limit: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    notes_df = pd.read_csv(notes_csv)
    if limit is not None:
        notes_df = notes_df.head(limit).copy()
    rows_by_note = group_rows_by_note(read_jsonl(reranked_jsonl))
    code2entry = load_corpus(corpus_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    previous: dict[str, dict[str, Any]] = {}
    if resume and output_csv.exists():
        prev_df = pd.read_csv(output_csv)
        if "note_id" in prev_df.columns and "final_predict" in prev_df.columns:
            for _, row in prev_df.iterrows():
                pred = row.get("final_predict")
                error = str(row.get("error") or "").strip()
                if isinstance(pred, str) and pred.strip() and not error:
                    previous[str(row["note_id"])] = row.to_dict()

    note_rows = notes_df.to_dict("records")
    outputs: list[dict[str, Any] | None] = [None] * len(note_rows)
    pending: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(note_rows):
        note_id = str(row.get("note_id") or "")
        if note_id in previous:
            outputs[idx] = previous[note_id]
        else:
            pending.append((idx, row))

    def flush() -> None:
        pd.DataFrame([item for item in outputs if item is not None]).to_csv(output_csv, index=False)

    kwargs = dict(
        code2entry=code2entry,
        model=model,
        max_candidates=max_candidates,
        max_tokens=max_tokens,
        json_mode=json_mode,
        stream=stream,
        use_neo4j_rules=use_neo4j_rules,
        sectionize=sectionize,
    )
    if max_concurrency <= 1:
        for done_count, (idx, row) in enumerate(tqdm(pending, desc="verifier"), start=1):
            outputs[idx] = _verify_one(row, rows_by_note.get(str(row.get("note_id")), []), **kwargs)
            if done_count % 2 == 0:
                flush()
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(_verify_one, row, rows_by_note.get(str(row.get("note_id")), []), **kwargs): idx
                for idx, row in pending
            }
            done_count = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="verifier"):
                idx = futures[future]
                outputs[idx] = future.result()
                done_count += 1
                if done_count % 2 == 0:
                    flush()
    flush()
    return pd.DataFrame([item for item in outputs if item is not None])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify note-level ICD-10-CM candidates with an LLM.")
    parser.add_argument("--notes-csv", default=str(config.OUTPUTS_DIR / "mdace_cm_discharge_notes.csv"))
    parser.add_argument("--reranked-jsonl", default=str(config.OUTPUTS_DIR / "reranked_candidates_cm.jsonl"))
    parser.add_argument("--corpus", default=str(config.ICD_CM_CORPUS_PATH))
    parser.add_argument("--output-csv", default=str(config.OUTPUTS_DIR / "final_predictions_cm.csv"))
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-sectionize", action="store_true")
    rule_group = parser.add_mutually_exclusive_group()
    rule_group.add_argument("--use-neo4j-rules", dest="use_neo4j_rules", action="store_true", default=False)
    rule_group.add_argument("--no-neo4j-rules", dest="use_neo4j_rules", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = run_verifier(
        args.notes_csv,
        args.reranked_jsonl,
        args.corpus,
        args.output_csv,
        model=args.model,
        max_candidates=args.max_candidates,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        json_mode=not args.no_json_mode,
        stream=args.stream,
        use_neo4j_rules=args.use_neo4j_rules,
        sectionize=not args.no_sectionize,
        limit=args.limit,
        resume=not args.no_resume,
    )
    ok = int((df.get("error", "").astype(str).str.len() == 0).sum()) if "error" in df.columns else len(df)
    print(f"Wrote {len(df)} rows to {args.output_csv}; successful rows: {ok}")


if __name__ == "__main__":
    main()
