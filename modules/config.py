from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:

    # 예시: 임베딩 모델명 (너 프로젝트에서 실제 쓰는 값으로 바꾸면 됨)
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

    # 예시: 청킹 파라미터
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # 예시: 평가 관련
    top_k: int = int(os.getenv("TOP_K", "5"))
