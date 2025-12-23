from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    """
    로컬 실행 확인용 더미 임베딩 (LangChain Embeddings 인터페이스 준수)
    - 같은 text -> 항상 같은 벡터(결정적)
    - torch/transformers 없이 동작
    """

    def __init__(self, dim: int = 1024, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    def _seed_from_text(self, text: str) -> int:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _embed_one(self, text: str) -> List[float]:
        seed = self._seed_from_text(text)
        rng = np.random.default_rng(seed)

        v = rng.standard_normal(self.dim).astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm

        return v.tolist()

    # ✅ LangChain이 요구하는 메서드 시그니처
    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]
