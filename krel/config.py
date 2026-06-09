from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_dotenv(path: str | Path = ROOT_DIR / ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

DATA_DIR = Path(os.getenv("KREL_DATA_DIR", ROOT_DIR / "data"))
OUTPUTS_DIR = Path(os.getenv("KREL_OUTPUTS_DIR", ROOT_DIR / "outputs"))

MDACE_DIR = DATA_DIR / "mdace"
ICD_CODES_DIR = DATA_DIR / "icd_codes"
ICD_EMBEDDINGS_DIR = DATA_DIR / "icd_embeddings"
ICD_CM_CORPUS_PATH = DATA_DIR / "icd_rag_corpus_augmented.jsonl"
ICD_CM_EMBEDDING_PATH = ICD_EMBEDDINGS_DIR / "icd_embeddings_qwen3_disease_augmented.npy"

LLM_API_KEY = os.getenv("KREL_LLM_API_KEY", "")
LLM_MODEL = os.getenv("KREL_LLM_MODEL", "gpt-4o")

EMBEDDING_MODEL = os.getenv("KREL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
RERANK_MODEL = os.getenv("KREL_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")

NEO4J_URI = os.getenv("KREL_NEO4J_URI", "")
NEO4J_DB = os.getenv("KREL_NEO4J_DB", "neo4j")
NEO4J_USER = os.getenv("KREL_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("KREL_NEO4J_PASSWORD", "")
