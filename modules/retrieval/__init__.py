from .retriever import search, get_vectorstore, RetrieverSettings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever():
    # qdrant 연결 + embeddings 생성 1회
    ...
    return retriever

__all__ = ["search", "get_vectorstore", "RetrieverSettings"]
