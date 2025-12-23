from __future__ import annotations

from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


def build_qdrant_vectorstore(
    *,
    documents: list[Document],
    embeddings,
    qdrant_path: Path,
    collection_name: str,
    recreate: bool = True,
    batch_size: int = 128,
):
    """
    documents -> (임베딩 생성) -> Qdrant 저장

    - recreate=True면 매번 컬렉션을 지우고 새로 만듦(개발/실험에 편함)
    """
    qdrant_path = Path(qdrant_path)
    qdrant_path.parent.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(qdrant_path))

    # 임베딩 차원 확인(한 번만)
    dim = len(embeddings.embed_query("임베딩 차원 확인"))

    if recreate:
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # 배치 삽입
    for i in range(0, len(documents), batch_size):
        store.add_documents(documents[i : i + batch_size])

    return store
