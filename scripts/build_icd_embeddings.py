from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_TASK = (
    "You are retrieving ICD-10-CM concept descriptions. Given a short disease or clinical concept "
    "description, return the most semantically matching ICD-10-CM concept descriptions."
)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def format_instruction(text: str, task: str = DEFAULT_TASK) -> str:
    return f"Instruct: {task}\nQuery: {text}"


def load_corpus_texts(corpus_path: Path) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            code = metadata.get("code") or row.get("code")
            if not code or code in seen:
                continue
            seen.add(str(code))
            text = str(row.get("page_content") or row.get("description") or code)
            codes.append(str(code))
            texts.append(format_instruction(text))
    return codes, texts


@torch.no_grad()
def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    dtype: str,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if torch.cuda.is_available() and dtype in {"float16", "fp16"}:
        model_kwargs["torch_dtype"] = torch.float16
    elif torch.cuda.is_available() and dtype in {"bfloat16", "bf16"}:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
    if not torch.cuda.is_available():
        model = model.to(device)

    arrays: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode ICD corpus"):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        arrays.append(embeddings.float().cpu().numpy())
    return np.concatenate(arrays, axis=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ICD-10-CM corpus embeddings for KREL recall.")
    parser.add_argument("--corpus", default="data/icd_rag_corpus_augmented.jsonl")
    parser.add_argument("--output", default="data/icd_embeddings/icd_embeddings_qwen3_disease_augmented.npy")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dtype", choices=["float32", "float16", "fp16", "bfloat16", "bf16"], default="float16")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    codes, texts = load_corpus_texts(corpus_path)
    if not texts:
        raise ValueError(f"No corpus texts loaded from {corpus_path}")
    embeddings = encode_texts(
        texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )
    if embeddings.shape[0] != len(codes):
        raise RuntimeError(f"Embedding rows ({embeddings.shape[0]}) do not match corpus codes ({len(codes)})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings.astype(np.float32))
    print(f"Wrote {embeddings.shape} embeddings to {output_path}")


if __name__ == "__main__":
    main()
