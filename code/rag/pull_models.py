from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from env_utils import HF_ENDPOINT, HF_HOME, RERANKER_MODEL


EMBEDDING_MODEL = "BAAI/bge-m3"


def main() -> None:
    print(f"HF_ENDPOINT={HF_ENDPOINT}")
    print(f"HF_HOME={HF_HOME or '<default>'}")
    print(f"EMBEDDING_MODEL={EMBEDDING_MODEL}")
    print(f"RERANKER_MODEL={RERANKER_MODEL}")

    cache_dir = None
    if HF_HOME:
        cache_dir = Path(HF_HOME).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resolved cache dir: {cache_dir}")

    embedding = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(cache_dir) if cache_dir else None,
    )
    print(f"SentenceTransformer loaded: {type(embedding).__name__}")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        cache_folder=str(cache_dir) if cache_dir else None,
    )
    print(f"CrossEncoder loaded: {type(reranker).__name__}")


if __name__ == "__main__":
    main()
