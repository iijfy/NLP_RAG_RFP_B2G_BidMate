# modules/retrieval/retriever.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from modules.paths import ProjectPaths
from modules.embedding.embedder import get_embeddings


@dataclass(frozen=True)
class RetrieverSettings:
    """
    Retriever가 참조하는 설정값들.
    - 로컬(dummy) / GCP(hf) 모두 "환경변수"만 바꿔서 동작하게 만드는 게 핵심.
    """
    mode: str = os.getenv("RAG_MODE", "recursive")  # recursive | semantic
    k: int = int(os.getenv("TOP_K", "5"))

    # ✅ 컬렉션 이름은 절대 하드코딩하지 말고 env로 받자
    #    로컬: rfp_recursive_DUMMY
    #    GCP : rfp_recursive_bge_m3 (예시)
    collection_name: str = os.getenv("QDRANT_COLLECTION", "rfp_recursive_DUMMY")

    # Qdrant는 로컬 파일 DB로 쓰고 있으니 path만 고정
    qdrant_dir_name: str = os.getenv("QDRANT_DIR", "qdrant_db")


def get_qdrant_path(settings: RetrieverSettings) -> Path:
    paths = ProjectPaths()
    return Path(paths.outputs_dir) / settings.qdrant_dir_name


def get_vectorstore(
    *,
    collection_name: Optional[str] = None,
    qdrant_path: Optional[Path] = None,
) -> QdrantVectorStore:
    """
    VectorStore 생성 함수.
    - 다른 모듈(Gradio/Eval/RAG chain)에서 store 만드는 방식을 통일해줌.
    """
    settings = RetrieverSettings()

    qdrant_path = qdrant_path or get_qdrant_path(settings)
    collection_name = collection_name or settings.collection_name

    client = QdrantClient(path=str(qdrant_path))
    embeddings = get_embeddings()  # dummy/hf 자동 선택

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def search(
    query: str,
    *,
    k: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> List[Document]:
    """
    ✅ 앞으로 검색은 무조건 이 함수만 쓰자.
    """
    settings = RetrieverSettings()
    k = k if k is not None else settings.k

    store = get_vectorstore(collection_name=collection_name)
    return store.similarity_search(query, k=k)
