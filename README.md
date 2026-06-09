# KREL

KREL is a minimal implementation of the MDACE ICD-10-CM full-label-space pipeline.

Main workflow:

1. Preprocess MDACE discharge summaries.
2. Extract ICD-oriented clinical queries with an OpenAI GPT model.
3. Retrieve candidate ICD-10-CM codes with hierarchy-aware beam search (HBS).
4. Rerank candidates with Qwen3-Reranker.
5. Verify note-level candidates with an OpenAI GPT model under a 50-code budget.
6. Evaluate micro precision, recall, and F1.

Neo4j rule augmentation is optional and disabled by default.

## Data

Bundled files:

- `data/mdace/`: MDACE ICD-10-CM train/validation/test CSV files.
- `data/aci_bench/`: ACI-BENCH train/valid/test JSONL files with dialogue, note, and ICD-10-CM codes.
- `data/icd_rag_corpus_augmented.jsonl`: ICD-10-CM retrieval corpus.
- `data/icd_codes/`: ICD-10-CM 2022 code list and tabular XML.
- `data/icd_embeddings/`: output directory for generated Qwen3 ICD embeddings.
- `simple_icd_10_cm/`: patched vendored ICD helper package used by recall and KG construction.

The default MDACE test setup filters `note_type == "Discharge summary"`, yielding 61 notes. The current main pipeline targets this MDACE ICD-10-CM full-label-space setting.

ACI-BENCH is included as an additional benchmark dataset. The bundled JSONL splits contain 67 train, 20 validation, and 120 test encounters.

Original MIMIC-III/MIMIC-IV records are controlled-access clinical datasets. If you need to use raw MIMIC data beyond the bundled public derivatives, apply for official access through PhysioNet and follow the required data-use agreements.

## Setup

```bash
cd KREL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```text
KREL_LLM_API_KEY=
KREL_LLM_MODEL=gpt-4o
KREL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
KREL_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
KREL_NEO4J_URI=
KREL_NEO4J_DB=neo4j
KREL_NEO4J_USER=neo4j
KREL_NEO4J_PASSWORD=
```

`KREL_LLM_API_KEY` is used for OpenAI Chat Completions. The CLI `--max-tokens` value is sent as `max_completion_tokens`.

## Run

Full pipeline:

```bash
python -m krel.pipeline --run-id mdace_cm_discharge_gpt4o --label-space full
```

Wrapper:

```bash
bash scripts/run_mdace_cm_discharge.sh
```

Outputs are written to:

```text
outputs/runs/<run-id>/
```

## Stages

Preprocess:

```bash
python -m krel.preprocess \
  --input data/mdace/mdace_icd10cm_test.csv \
  --output outputs/mdace_cm_discharge_notes.csv
```

Query extraction:

```bash
python -m krel.query_retrieval \
  --notes-csv outputs/mdace_cm_discharge_notes.csv \
  --output-csv outputs/mdace_cm_discharge_queries.csv \
  --model gpt-4o \
  --max-tokens 32768
```

HBS recall:

```bash
python scripts/build_icd_embeddings.py \
  --corpus data/icd_rag_corpus_augmented.jsonl \
  --output data/icd_embeddings/icd_embeddings_qwen3_disease_augmented.npy \
  --model Qwen/Qwen3-Embedding-8B
```

```bash
python -m krel.recall \
  --queries-csv outputs/mdace_cm_discharge_queries.csv \
  --output-jsonl outputs/query_candidates_cm.jsonl \
  --results-csv outputs/recall_results_cm.csv \
  --label-space full
```

Rerank:

```bash
python -m krel.rerank \
  --query-candidates outputs/query_candidates_cm.jsonl \
  --output-jsonl outputs/reranked_candidates_cm.jsonl \
  --metrics-json outputs/rerank_metrics_cm.json \
  --topk 50
```

Verifier:

```bash
python -m krel.verifier \
  --notes-csv outputs/mdace_cm_discharge_notes.csv \
  --reranked-jsonl outputs/reranked_candidates_cm.jsonl \
  --output-csv outputs/final_predictions_cm.csv \
  --max-candidates 50 \
  --max-tokens 32768
```

Evaluate:

```bash
python -m krel.evaluate \
  --predictions-csv outputs/final_predictions_cm.csv \
  --output-json outputs/evaluation_metrics_cm.json
```

## Optional Neo4j Rules

The base pipeline does not require Neo4j. To enable rule hints and combination checks:

```bash
python -m krel.kg_builder
python -m krel.pipeline --run-id mdace_cm_discharge_gpt4o_rules --label-space full --use-neo4j-rules
```

Pairwise rule hints only reference codes already present in the note-level candidate set. Combination targets may be added only through explicit Stage 2 combination checks.

## Dataset Sources and Citations

MDACE data was copied from https://github.com/3mcloud/MDACE/tree/main. Please cite:

```bibtex
@inproceedings{cheng2023mdace,
  title = {{MDACE}: {MIMIC} Documents Annotated with Code Evidence},
  author = {Cheng, Hua and Jafari, Rana and Russell, April and Klopfer, Russell and Lu, Edmond and Striner, Benjamin and Gormley, Matthew},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2023},
  pages = {7534--7550},
  url = {https://aclanthology.org/2023.acl-long.416}
}
```

ACI-BENCH data was copied from https://github.com/wyim/aci-bench. The dataset is released under CC BY 4.0. Please cite:

```bibtex
@article{yim2023acibench,
  title = {Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  author = {Yim, Wen-wai and Fu, Yujuan and Ben Abacha, Asma and Snider, Neal and Lin, Thomas and Yetisgen, Meliha},
  journal = {Scientific Data},
  volume = {10},
  number = {1},
  pages = {586},
  year = {2023},
  doi = {10.1038/s41597-023-02487-3}
}
```

## Notes

- HBS defaults: `Kc=200`, `B=200`, `M=8`, `D=6`, `Kf=30`.
- The verifier keeps at most 50 unique candidates per note.
- Recall requires `data/icd_embeddings/icd_embeddings_qwen3_disease_augmented.npy`; generate it with `scripts/build_icd_embeddings.py`.
- Embedding generation and reranking require Hugging Face model downloads unless cached.
- A GPU is strongly recommended for recall and reranking.
