from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import simple_icd_10_cm as cm
from tqdm import tqdm

from . import config


RULE_REL_GETTERS = {
    "useAdditionalCode": cm.get_use_additional_code,
    "codeFirst": cm.get_code_first,
    "codeAlso": cm.get_code_also,
}


@dataclass(frozen=True)
class CombinationRule:
    rule_id: str
    target_code: str
    components: tuple[str, ...]
    source: str = "manual_extension"
    logic: str = "AND"
    allow_descendant_match: bool = True
    note: str = ""


DEFAULT_COMBINATION_RULES: tuple[CombinationRule, ...] = (
    CombinationRule(
        rule_id="comb_I129_htn_ckd",
        target_code="I12.9",
        components=("I10", "N18"),
        note="Hypertensive chronic kidney disease with CKD stage 1-4/unspecified.",
    ),
    CombinationRule(
        rule_id="comb_I110_htn_hf",
        target_code="I11.0",
        components=("I10", "I50"),
        note="Hypertensive heart disease with heart failure.",
    ),
    CombinationRule(
        rule_id="comb_I130_htn_hf_ckd",
        target_code="I13.0",
        components=("I10", "I50", "N18"),
        note="Hypertensive heart and CKD with heart failure and CKD stage 1-4/unspecified.",
    ),
    CombinationRule(
        rule_id="comb_E1122_dm_ckd",
        target_code="E11.22",
        components=("E11", "N18"),
        note="Type 2 diabetes mellitus with diabetic chronic kidney disease.",
    ),
)


def require_neo4j_config() -> None:
    if not config.NEO4J_URI:
        raise RuntimeError("Missing KREL_NEO4J_URI. Set Neo4j connection fields in .env first.")
    if not config.NEO4J_PASSWORD:
        raise RuntimeError("Missing KREL_NEO4J_PASSWORD. Set Neo4j connection fields in .env first.")


def make_driver():
    from neo4j import GraphDatabase

    require_neo4j_config()
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def load_cm_tree(codes_file: str | Path, tabular_file: str | Path) -> None:
    cm.change_version(
        all_codes_file_path=str(codes_file),
        classification_data_file_path=str(tabular_file),
    )


def chunks(values: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def get_joined_str(func, code: str) -> str:
    try:
        values = func(code) or []
    except Exception:
        values = []
    return "; ".join(str(value) for value in values if str(value).strip())


def normalize_target_code(code_str: str) -> str:
    code_str = str(code_str).strip()
    if code_str.endswith(".-"):
        return code_str[:-2]
    if code_str.endswith("-"):
        return code_str[:-1]
    return code_str


def get_peer_codes_in_range(start_code: str, end_code: str, all_codes: list[str]) -> list[str]:
    results: list[str] = []
    target_len = len(start_code)
    end_boundary = end_code + "~"
    for code in all_codes:
        if code < start_code:
            continue
        if code > end_boundary:
            break
        if len(code) == target_len:
            results.append(code)
    return results


def parse_targets(text: str, all_codes: list[str]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    regex_code = r"[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?(?:[\.\-]+)?"
    regex_range = rf"({regex_code})\s*(?:to|--|-)\s*({regex_code})"
    temp_text = str(text or "")

    for match in re.finditer(regex_range, temp_text):
        full_match = match.group(0)
        start = normalize_target_code(match.group(1))
        end = normalize_target_code(match.group(2))
        if cm.is_block(f"{start}-{end}"):
            targets.append({"type": "single_code", "original": full_match, "code": f"{start}-{end}"})
        else:
            for peer_code in get_peer_codes_in_range(start, end, all_codes):
                targets.append({"type": "single_code", "original": full_match, "code": peer_code})
        temp_text = temp_text.replace(full_match, " ")

    for match in re.finditer(rf"\b({regex_code})\b", temp_text):
        raw_code = match.group(1)
        clean_code = normalize_target_code(raw_code)
        if clean_code:
            targets.append({"type": "single_code", "original": raw_code, "code": clean_code})
    return targets


def create_constraints(session) -> None:
    session.run("CREATE CONSTRAINT icd_code_code IF NOT EXISTS FOR (c:ICD_Code) REQUIRE c.code IS UNIQUE")
    session.run("CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE")


def merge_icd_nodes(session, rows: list[dict[str, Any]]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:ICD_Code {code: row.code})
        SET c.description = row.description,
            c.includes = row.includes,
            c.inclusion_term = row.inclusion_term,
            c.version = row.version
        """,
        rows=rows,
    )


def merge_relationships(session, rel_type: str, rows: list[dict[str, Any]]) -> None:
    if rel_type not in {"IS_A", "useAdditionalCode", "codeFirst", "codeAlso"}:
        raise ValueError(f"Unsupported relationship type: {rel_type}")
    session.run(
        f"""
        UNWIND $rows AS row
        MERGE (source:ICD_Code {{code: row.source}})
        MERGE (target:ICD_Code {{code: row.target}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.original_text = coalesce(row.original_text, r.original_text, "")
        """,
        rows=rows,
    )


def merge_combination_rules(session, rules: Iterable[CombinationRule]) -> None:
    for rule in rules:
        session.run(
            """
            MERGE (r:Rule {id: $rule_id})
            SET r.type = 'combination',
                r.target_code = $target_code,
                r.logic = $logic,
                r.source = $source,
                r.allow_descendant_match = $allow_descendant_match,
                r.note = $note
            MERGE (target:ICD_Code {code: $target_code})
            MERGE (r)-[:YIELDS]->(target)
            WITH r
            UNWIND $components AS component
            MERGE (c:ICD_Code {code: component.code})
            MERGE (r)-[hc:HAS_COMPONENT]->(c)
            SET hc.order = component.order,
                hc.match_mode = component.match_mode
            """,
            rule_id=rule.rule_id,
            target_code=rule.target_code,
            logic=rule.logic,
            source=rule.source,
            allow_descendant_match=rule.allow_descendant_match,
            note=rule.note,
            components=[
                {
                    "code": component,
                    "order": idx,
                    "match_mode": "self_or_descendant" if rule.allow_descendant_match else "exact",
                }
                for idx, component in enumerate(rule.components, start=1)
            ],
        )


def build_kg(
    *,
    codes_file: str | Path,
    tabular_file: str | Path,
    version: str = "2022",
    batch_size: int = 1000,
    skip_nodes: bool = False,
    skip_hierarchy: bool = False,
    skip_pairwise_rules: bool = False,
    skip_combination_rules: bool = False,
) -> None:
    load_cm_tree(codes_file, tabular_file)
    all_codes = sorted(cm.get_all_codes(with_dots=True))
    driver = make_driver()
    with driver.session(database=config.NEO4J_DB) as session:
        create_constraints(session)

        if not skip_nodes:
            node_rows = [
                {
                    "code": code,
                    "description": cm.get_description(code) or "",
                    "includes": get_joined_str(cm.get_includes, code),
                    "inclusion_term": get_joined_str(cm.get_inclusion_term, code),
                    "version": version,
                }
                for code in tqdm(all_codes, desc="prepare ICD nodes")
            ]
            for batch in tqdm(list(chunks(node_rows, batch_size)), desc="merge ICD nodes"):
                merge_icd_nodes(session, batch)

        if not skip_hierarchy:
            hierarchy_rows: list[dict[str, Any]] = []
            for code in tqdm(all_codes, desc="prepare hierarchy edges"):
                try:
                    parent = cm.get_parent(code)
                except Exception:
                    parent = None
                if parent:
                    hierarchy_rows.append({"source": code, "target": parent, "original_text": ""})
            for batch in tqdm(list(chunks(hierarchy_rows, batch_size)), desc="merge IS_A edges"):
                merge_relationships(session, "IS_A", batch)

        if not skip_pairwise_rules:
            for rel_type, getter in RULE_REL_GETTERS.items():
                edge_rows: list[dict[str, Any]] = []
                for code in tqdm(all_codes, desc=f"prepare {rel_type} edges"):
                    rule_text = get_joined_str(getter, code)
                    if not rule_text:
                        continue
                    for target in parse_targets(rule_text, all_codes):
                        edge_rows.append(
                            {
                                "source": code,
                                "target": target["code"],
                                "original_text": target.get("original", ""),
                            }
                        )
                for batch in tqdm(list(chunks(edge_rows, batch_size)), desc=f"merge {rel_type} edges"):
                    merge_relationships(session, rel_type, batch)

        if not skip_combination_rules:
            merge_combination_rules(session, DEFAULT_COMBINATION_RULES)
    driver.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ICD-10-CM Neo4j knowledge graph used by KREL.")
    parser.add_argument("--codes-file", default=str(config.ICD_CODES_DIR / "icd10cm_codes_2022.txt"))
    parser.add_argument("--tabular-file", default=str(config.ICD_CODES_DIR / "icd10cm_tabular_2022.xml"))
    parser.add_argument("--version", default="2022")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--skip-nodes", action="store_true")
    parser.add_argument("--skip-hierarchy", action="store_true")
    parser.add_argument("--skip-pairwise-rules", action="store_true")
    parser.add_argument("--skip-combination-rules", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_kg(
        codes_file=args.codes_file,
        tabular_file=args.tabular_file,
        version=args.version,
        batch_size=args.batch_size,
        skip_nodes=args.skip_nodes,
        skip_hierarchy=args.skip_hierarchy,
        skip_pairwise_rules=args.skip_pairwise_rules,
        skip_combination_rules=args.skip_combination_rules,
    )


if __name__ == "__main__":
    main()
