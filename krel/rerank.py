from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .io_utils import normalize_code, read_jsonl, write_jsonl


class Qwen3Reranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        max_length: int = 256,
        use_flash_attn: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
        self.max_length = max_length

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.token_false_id is None or self.token_true_id is None:
            raise ValueError('Cannot find token ids for "yes"/"no". Check tokenizer vocab.')

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    @staticmethod
    def format_text(instruction: str, query: str, doc: str) -> str:
        instruction = instruction or "Given a query, retrieve relevant documents."
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in batch:
            batch[key] = batch[key].to(self.model.device)
        return batch

    @torch.no_grad()
    def score(self, formatted_pairs: list[str], batch_size: int = 16) -> list[float]:
        scores_all: list[float] = []
        for start in range(0, len(formatted_pairs), batch_size):
            batch_text = formatted_pairs[start : start + batch_size]
            inputs = self._process_inputs(batch_text)
            logits = self.model(**inputs).logits[:, -1, :]
            true_v = logits[:, self.token_true_id]
            false_v = logits[:, self.token_false_id]
            two = torch.stack([false_v, true_v], dim=1)
            logp = F.log_softmax(two, dim=1)
            scores_all.extend(logp[:, 1].exp().detach().cpu().tolist())
        return [float(value) for value in scores_all]

    def rerank(
        self,
        query: str,
        docs: list[str],
        instruction: str,
        topk: int,
        batch_size: int = 16,
    ) -> tuple[list[int], list[float]]:
        formatted = [self.format_text(instruction, query, doc) for doc in docs]
        scores = self.score(formatted, batch_size=batch_size)
        idx_score = list(enumerate(scores))
        idx_score.sort(key=lambda item: item[1], reverse=True)
        if topk is not None:
            idx_score = idx_score[: min(topk, len(idx_score))]
        return [i for i, _ in idx_score], [s for _, s in idx_score]


def load_kb_code2text(kb_jsonl: str | Path) -> dict[str, str]:
    code2text: dict[str, str] = {}
    for obj in read_jsonl(kb_jsonl):
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        code = metadata.get("code") or obj.get("code")
        if not code:
            continue
        text = str(obj.get("page_content") or obj.get("description") or code)
        code2text.setdefault(str(code), text)
        code2text.setdefault(normalize_code(code), text)
    return code2text


def compute_query_metrics(rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 5, 10, 20, 50)) -> dict[str, Any]:
    totals = {k: {"recall_sum": 0.0, "hit_sum": 0.0, "n": 0} for k in ks}
    for row in rows:
        gt_set = {normalize_code(code) for code in (row.get("gt_codes") or [])}
        if not gt_set:
            continue
        ranked = [normalize_code(code) for code in (row.get("reranked_codes") or [])]
        for k in ks:
            topk = set(ranked[: min(k, len(ranked))])
            hit_count = len(gt_set & topk)
            totals[k]["recall_sum"] += hit_count / len(gt_set)
            totals[k]["hit_sum"] += 1.0 if hit_count else 0.0
            totals[k]["n"] += 1

    metrics: dict[str, Any] = {}
    for k in ks:
        n = totals[k]["n"]
        metrics[f"recall@{k}"] = None if n == 0 else totals[k]["recall_sum"] / n
        metrics[f"hit@{k}"] = None if n == 0 else totals[k]["hit_sum"] / n
    metrics["num_queries_evaluated"] = max(totals[ks[0]]["n"], 0)
    return metrics


def run_rerank(
    query_candidates: str | Path,
    corpus_path: str | Path,
    output_jsonl: str | Path,
    metrics_json: str | Path,
    *,
    model_name: str = config.RERANK_MODEL,
    topk: int = 50,
    batch_size: int = 16,
    max_length: int = 256,
    truncate_doc_chars: int = 1200,
    use_flash_attn: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(query_candidates)
    code2text = load_kb_code2text(corpus_path)
    reranker = Qwen3Reranker(model_name=model_name, max_length=max_length, use_flash_attn=use_flash_attn)
    instruction = (
        "Given an ICD-oriented clinical query, retrieve the most relevant ICD-10-CM code description. "
        "Prefer exact diagnosis/status matches and documented specificity."
    )

    output_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="rerank"):
        query = str(row.get("query") or "")
        candidates = [str(code) for code in (row.get("candidates") or [])]
        docs = []
        for code in candidates:
            doc = code2text.get(code) or code2text.get(normalize_code(code)) or code
            if truncate_doc_chars and truncate_doc_chars > 0:
                doc = doc[:truncate_doc_chars]
            docs.append(doc)
        out = dict(row)
        if not candidates:
            out["reranked_codes"] = []
            out["reranked_scores"] = []
            output_rows.append(out)
            continue
        top_idx, scores = reranker.rerank(
            query=query,
            docs=docs,
            instruction=instruction,
            topk=topk,
            batch_size=batch_size,
        )
        out["reranked_codes"] = [candidates[i] for i in top_idx]
        out["reranked_scores"] = scores
        output_rows.append(out)

    write_jsonl(output_jsonl, output_rows)
    metrics = compute_query_metrics(output_rows)
    metrics["topk_kept"] = topk
    metrics["model"] = model_name
    metrics_json = Path(metrics_json)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank ICD-10-CM recall candidates with a Qwen3 reranker.")
    parser.add_argument("--query-candidates", default=str(config.OUTPUTS_DIR / "query_candidates_cm.jsonl"))
    parser.add_argument("--corpus", default=str(config.ICD_CM_CORPUS_PATH))
    parser.add_argument("--output-jsonl", default=str(config.OUTPUTS_DIR / "reranked_candidates_cm.jsonl"))
    parser.add_argument("--metrics-json", default=str(config.OUTPUTS_DIR / "rerank_metrics_cm.json"))
    parser.add_argument("--model", default=config.RERANK_MODEL)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--truncate-doc-chars", type=int, default=1200)
    parser.add_argument("--use-flash-attn", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_rerank(
        args.query_candidates,
        args.corpus,
        args.output_jsonl,
        args.metrics_json,
        model_name=args.model,
        topk=args.topk,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_doc_chars=args.truncate_doc_chars,
        use_flash_attn=args.use_flash_attn,
    )


if __name__ == "__main__":
    main()
