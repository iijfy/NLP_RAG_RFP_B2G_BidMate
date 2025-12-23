# modules/rag/chain.py
from __future__ import annotations

from modules.retrieval import search
from modules.generator.llm import generate_answer


def answer(question: str, *, k: int = 5) -> str:
    """
    RAG의 핵심 함수: 질문 -> 검색 -> 답변 생성
    """
    docs = search(question, k=k)
    return generate_answer(question, docs)
