from __future__ import annotations

import os
from typing import Optional


def get_embeddings(*, backend: Optional[str] = None):
    """
    backend:
      - "dummy": 로컬 실행 확인용 (torch/transformers 불필요)
      - "hf": HuggingFace (GCP에서 bge-m3 등 실제 임베딩)
    """
    backend = backend or os.getenv("EMBEDDINGS_BACKEND", "dummy").lower()

    if backend == "dummy":
        from modules.embedding.backends.dummy import DummyEmbeddings
        dim = int(os.getenv("DUMMY_EMBEDDING_DIM", "1024"))
        print(f"[Embeddings] backend=dummy dim={dim}")
        return DummyEmbeddings(dim=dim, normalize=True)

    if backend == "hf":
        # ⚠️ GCP에서만 쓰는 걸 권장 (로컬은 torch 이슈가 있으니)
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")

        print(f"[Embeddings] backend=hf model={model_name} device={device}")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown EMBEDDINGS_BACKEND: {backend}")
