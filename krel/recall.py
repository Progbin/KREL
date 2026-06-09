from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simple_icd_10_cm as cm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config
from .io_utils import extract_query_items, normalize_code, parse_ground_truth, write_jsonl


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def to_category(code: str) -> str | None:
    match = re.match(r"^([A-Z][0-9]{2})", str(code).strip().upper())
    return match.group(1) if match else None


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def norm_to_cm_code(norm_code: str) -> str:
    code = str(norm_code).strip().upper()
    if not code or "-" in code or "." in code or len(code) <= 3:
        return code
    return code[:3] + "." + code[3:]


def load_observed_code_set(dataframes: list[pd.DataFrame], col: str = "ground_truth") -> set[str]:
    observed: set[str] = set()
    for dataframe in dataframes:
        if dataframe is None or dataframe.empty or col not in dataframe.columns:
            continue
        for value in dataframe[col].values:
            observed.update(parse_ground_truth(value))
    return observed


def load_cm_tree(codes_file: str | Path, tabular_file: str | Path) -> None:
    cm.change_version(
        all_codes_file_path=str(codes_file),
        classification_data_file_path=str(tabular_file),
    )


class SingleIndexBeamRetriever:
    def __init__(
        self,
        *,
        query_model_name: str,
        allcode_emb_path: str | Path,
        kb_corpus_path: str | Path,
        max_length: int = 256,
        top_k_category: int = 200,
        beam_size: int = 200,
        per_parent_topm: int = 8,
        max_depth: int = 6,
        final_candidate_budget: int = 30,
        allowed_norm_codes: set[str] | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.top_k_category = top_k_category
        self.beam_size = beam_size
        self.per_parent_topm = per_parent_topm
        self.max_depth = max_depth
        self.final_candidate_budget = final_candidate_budget
        self.allowed_norm_codes: set[str] | None = None
        self.allowed_categories: set[str] | None = None
        self.allowed_tree_nodes: set[str] | None = None

        if allowed_norm_codes:
            self.allowed_norm_codes = {normalize_code(code) for code in allowed_norm_codes}
            categories = {cat for code in self.allowed_norm_codes if (cat := to_category(code))}
            self.allowed_categories = categories or None
            tree_nodes: set[str] = set()
            for norm in self.allowed_norm_codes:
                current = norm_to_cm_code(norm)
                while current:
                    tree_nodes.add(current)
                    try:
                        if cm.is_category(current):
                            break
                        current = cm.get_parent(current)
                    except Exception:
                        break
            self.allowed_tree_nodes = tree_nodes or None

        print(f"Loading query encoder on {self.device}: {query_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(query_model_name, padding_side="left")
        model_kwargs: dict[str, Any] = {"device_map": "auto"} if torch.cuda.is_available() else {}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(query_model_name, **model_kwargs)
        self.model.eval()
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        print(f"Loading all-code embeddings: {allcode_emb_path}")
        all_embeddings = np.load(allcode_emb_path)
        self.all_embs = torch.as_tensor(all_embeddings).to(self.device)
        if torch.cuda.is_available() and getattr(self.model, "dtype", None) == torch.float16:
            self.all_embs = self.all_embs.to(dtype=torch.float16)
        self.all_embs = F.normalize(self.all_embs, p=2, dim=1)

        print(f"Loading KB corpus index: {kb_corpus_path}")
        self.code2idx: dict[str, int] = {}
        self.idx2code: list[str] = []
        self.category_indices: list[int] = []
        row_counter = 0
        with Path(kb_corpus_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                code = item["metadata"]["code"]
                if code in self.code2idx:
                    continue
                self.code2idx[code] = row_counter
                self.idx2code.append(code)
                if cm.is_category(code):
                    self.category_indices.append(row_counter)
                row_counter += 1
        if len(self.idx2code) != self.all_embs.shape[0]:
            print(
                "Warning: corpus unique codes "
                f"({len(self.idx2code)}) != embedding rows ({self.all_embs.shape[0]}). "
                "The embedding file must use the same unique-code order as the corpus."
            )

        self.task_instruction = (
            "You are retrieving ICD-10-CM concept descriptions. Given a short disease or clinical concept "
            "description, return the most semantically matching ICD-10-CM concept descriptions."
        )

    def encode_query(self, text: str) -> torch.Tensor:
        instructed_text = get_detailed_instruct(self.task_instruction, clean_text(text))
        batch = self.tokenizer(
            [instructed_text],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
            embedding = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def _score_codes(self, query_embedding: torch.Tensor, codes: list[str]) -> tuple[list[str], torch.Tensor]:
        indices: list[int] = []
        kept: list[str] = []
        for code in codes:
            index = self.code2idx.get(code)
            if index is None:
                continue
            indices.append(index)
            kept.append(code)
        if not kept:
            return [], torch.empty(0, device=self.device)
        embeddings = self.all_embs[torch.tensor(indices, device=self.device)]
        scores = torch.mm(query_embedding, embeddings.t())[0]
        return kept, scores

    def retrieve_topk_categories_from_embedding(self, query_embedding: torch.Tensor, topk: int | None = None):
        if not self.category_indices:
            return []
        if self.allowed_categories:
            filtered = [idx for idx in self.category_indices if self.idx2code[idx] in self.allowed_categories]
            if not filtered:
                return []
            cat_idx_tensor = torch.tensor(filtered, device=self.device)
            cat_codes = [self.idx2code[i] for i in filtered]
        else:
            cat_idx_tensor = torch.tensor(self.category_indices, device=self.device)
            cat_codes = [self.idx2code[i] for i in self.category_indices]
        scores = torch.mm(query_embedding, self.all_embs[cat_idx_tensor].t())[0]
        k = min(topk or self.top_k_category, scores.numel())
        top_vals, top_inds = torch.topk(scores, k=k)
        top_inds_np = top_inds.detach().cpu().numpy()
        top_vals_np = top_vals.detach().cpu().numpy()
        return [(cat_codes[int(i)], float(v)) for i, v in zip(top_inds_np, top_vals_np)]

    def _children(self, code: str) -> list[str]:
        try:
            children = list(cm.get_children(code) or [])
            if self.allowed_tree_nodes is not None:
                children = [child for child in children if child in self.allowed_tree_nodes]
            return children
        except Exception:
            return []

    def retrieve_beam_search(self, query_text: str, return_scores: bool = False):
        query_embedding = self.encode_query(query_text)
        entry = self.retrieve_topk_categories_from_embedding(query_embedding, topk=self.top_k_category)
        if not entry:
            return []

        beam = [(code, score, 0) for code, score in entry]
        final: dict[str, float] = {}
        visited: set[str] = set()

        for _ in range(self.max_depth):
            expanded = []
            for parent_code, parent_score, depth in beam:
                if parent_code in visited:
                    continue
                visited.add(parent_code)
                children = self._children(parent_code)
                if not children:
                    self._maybe_add_final(final, parent_code, parent_score)
                    continue
                kept_children, child_scores = self._score_codes(query_embedding, children)
                if not kept_children:
                    continue
                m = min(self.per_parent_topm, child_scores.numel())
                top_vals, top_inds = torch.topk(child_scores, k=m)
                top_inds_np = top_inds.detach().cpu().numpy()
                top_vals_np = top_vals.detach().cpu().numpy()
                for idx, local in zip(top_inds_np, top_vals_np):
                    child = kept_children[int(idx)]
                    new_score = 0.5 * float(parent_score) + 0.5 * float(local)
                    if cm.is_leaf(child) or len(self._children(child)) == 0:
                        self._maybe_add_final(final, child, new_score)
                    expanded.append((child, new_score, depth + 1))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[1], reverse=True)
            beam = expanded[: self.beam_size]

        items = sorted(final.items(), key=lambda kv: kv[1], reverse=True)[: self.final_candidate_budget]
        return items if return_scores else [code for code, _ in items]

    def _maybe_add_final(self, final: dict[str, float], code: str, score: float) -> None:
        normalized = normalize_code(code)
        if self.allowed_norm_codes is not None and normalized not in self.allowed_norm_codes:
            return
        previous = final.get(code)
        if previous is None or score > previous:
            final[code] = float(score)


def run_recall(
    queries_csv: str | Path,
    output_jsonl: str | Path,
    results_csv: str | Path,
    *,
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    corpus_path: str | Path = config.ICD_CM_CORPUS_PATH,
    embedding_path: str | Path = config.ICD_CM_EMBEDDING_PATH,
    codes_file: str | Path = config.ICD_CODES_DIR / "icd10cm_codes_2022.txt",
    tabular_file: str | Path = config.ICD_CODES_DIR / "icd10cm_tabular_2022.xml",
    query_model_name: str = config.EMBEDDING_MODEL,
    label_space: str = "full",
    max_length: int = 256,
    top_k_category: int = 200,
    beam_size: int = 200,
    per_parent_topm: int = 8,
    max_depth: int = 6,
    final_candidate_budget: int = 30,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    load_cm_tree(codes_file, tabular_file)
    df = pd.read_csv(queries_csv)
    train_df = pd.read_csv(train_csv) if train_csv else pd.DataFrame()
    val_df = pd.read_csv(val_csv) if val_csv else pd.DataFrame()
    allowed_norm_codes = None
    if label_space == "observed":
        allowed_norm_codes = load_observed_code_set([train_df, val_df, df], col="ground_truth")
        print(f"Observed label space size: {len(allowed_norm_codes)}")
    else:
        print("Full label space enabled.")

    retriever = SingleIndexBeamRetriever(
        query_model_name=query_model_name,
        allcode_emb_path=embedding_path,
        kb_corpus_path=corpus_path,
        max_length=max_length,
        top_k_category=top_k_category,
        beam_size=beam_size,
        per_parent_topm=per_parent_topm,
        max_depth=max_depth,
        final_candidate_budget=final_candidate_budget,
        allowed_norm_codes=allowed_norm_codes,
    )

    query_level_outputs: list[dict[str, Any]] = []
    note2cands: dict[str, set[str]] = defaultdict(set)
    note2gt: dict[str, set[str]] = {}

    for idx, row in tqdm(list(df.iterrows()), desc="recall"):
        note_id = str(row.get("note_id", idx))
        hadm_id = str(row.get("hadm_id", idx))
        gt_codes = parse_ground_truth(row.get("ground_truth"))
        note2gt[note_id] = gt_codes
        query_items = extract_query_items(row.get("respond"))
        for item in query_items:
            query = clean_text(str(item.get("query") or ""))
            if not query:
                continue
            scored = retriever.retrieve_beam_search(query, return_scores=True)
            pruned = [code for code, _ in scored]
            score_map = {code: float(score) for code, score in scored}
            note2cands[note_id].update(normalize_code(code) for code in pruned)
            query_level_outputs.append(
                {
                    "hadm_id": hadm_id,
                    "note_id": note_id,
                    "query": query,
                    "base": item.get("base"),
                    "evidence": item.get("evidence"),
                    "gt_codes": sorted(gt_codes),
                    "candidates": pruned,
                    "candidate_scores": [score_map.get(code, 0.0) for code in pruned],
                    "candidates_count": len(pruned),
                }
            )

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        note_id = str(row.get("note_id"))
        hadm_id = str(row.get("hadm_id"))
        gt = note2gt.get(note_id, parse_ground_truth(row.get("ground_truth")))
        cands = note2cands.get(note_id, set())
        hits = len(gt & cands)
        recall = hits / len(gt) if gt else 0.0
        precision = hits / len(cands) if cands else 0.0
        results.append(
            {
                "hadm_id": hadm_id,
                "note_id": note_id,
                "query_count": len(extract_query_items(row.get("respond"))),
                "candidates_count": len(cands),
                "hits": hits,
                "gt_count": len(gt),
                "recall": recall,
                "precision": precision,
            }
        )

    output_jsonl = Path(output_jsonl)
    results_csv = Path(results_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, query_level_outputs)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)
    if not results_df.empty:
        print(f"Mean recall: {results_df['recall'].mean():.4f}")
        print(f"Mean candidate set size: {results_df['candidates_count'].mean():.1f}")
    return results_df, query_level_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ICD-10-CM beam recall from extracted queries.")
    parser.add_argument("--queries-csv", default=str(config.OUTPUTS_DIR / "mdace_cm_discharge_queries.csv"))
    parser.add_argument("--output-jsonl", default=str(config.OUTPUTS_DIR / "query_candidates_cm.jsonl"))
    parser.add_argument("--results-csv", default=str(config.OUTPUTS_DIR / "recall_results_cm.csv"))
    parser.add_argument("--train-csv", default=str(config.MDACE_DIR / "mdace_icd10cm_train.csv"))
    parser.add_argument("--val-csv", default=str(config.MDACE_DIR / "mdace_icd10cm_validation.csv"))
    parser.add_argument("--corpus", default=str(config.ICD_CM_CORPUS_PATH))
    parser.add_argument("--embedding-path", default=str(config.ICD_CM_EMBEDDING_PATH))
    parser.add_argument("--codes-file", default=str(config.ICD_CODES_DIR / "icd10cm_codes_2022.txt"))
    parser.add_argument("--tabular-file", default=str(config.ICD_CODES_DIR / "icd10cm_tabular_2022.xml"))
    parser.add_argument("--query-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--label-space", choices=["full", "observed"], default="full")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-category", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--per-parent-topm", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--final-candidate-budget", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_recall(
        args.queries_csv,
        args.output_jsonl,
        args.results_csv,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        corpus_path=args.corpus,
        embedding_path=args.embedding_path,
        codes_file=args.codes_file,
        tabular_file=args.tabular_file,
        query_model_name=args.query_model,
        label_space=args.label_space,
        max_length=args.max_length,
        top_k_category=args.top_k_category,
        beam_size=args.beam_size,
        per_parent_topm=args.per_parent_topm,
        max_depth=args.max_depth,
        final_candidate_budget=args.final_candidate_budget,
    )


if __name__ == "__main__":
    main()
