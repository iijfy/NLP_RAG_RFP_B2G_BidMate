# modules/rag/pipeline.py
from __future__ import annotations
from typing import List, Tuple, Optional
from langchain_core.documents import Document

from modules.retrieval import search
from modules.rag.generator import generate_answer

def answer_query(query: str, k: int = 3, docs: Optional[List[Document]] = None) -> Tuple[str, List[Document]]:
    # docs가 들어오면 검색 재사용, 없으면 검색 수행
    if docs is None:
        docs = search(query, k=k)

    answer = generate_answer(query, docs)
    return answer, docs
