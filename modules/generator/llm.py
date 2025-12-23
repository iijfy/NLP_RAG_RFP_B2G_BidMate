# modules/generator/llm.py
from __future__ import annotations

import os
from typing import List
from langchain_core.documents import Document


def generate_answer_dummy(question: str, docs: List[Document]) -> str:
    """
    로컬 실행 확인용 더미 답변기.
    - 중요한 건 '체인이 끝까지 돈다'를 확인하는 것.
    - 문서 요약처럼 흉내만 냄.
    """
    lines = []
    lines.append("※ [DUMMY ANSWER] (로컬 실행 확인용)")
    lines.append(f"질문: {question}")
    lines.append("")
    lines.append("참고한 문서 Top-k 요약:")

    for i, d in enumerate(docs, 1):
        meta = d.metadata
        doc_id = meta.get("doc_id")
        chunk_id = meta.get("chunk_id")
        source = meta.get("source")
        snippet = d.page_content.replace("\n", " ")[:160]
        lines.append(f"- [{i}] doc_id={doc_id}, chunk_id={chunk_id}, source={source} :: {snippet}...")

    lines.append("")
    lines.append("최종 답변(더미): 위 문서들을 근거로 요구사항/조건/유지보수/네트워크 등 항목이 포함됩니다.")
    return "\n".join(lines)


def generate_answer(question: str, docs: List[Document]) -> str:
    """
    backend에 따라 더미/실제 LLM을 선택.
    - 로컬: dummy
    - GCP : openai/hf 등으로 확장 가능
    """
    backend = os.getenv("GENERATOR_BACKEND", "dummy").lower()

    if backend == "dummy":
        return generate_answer_dummy(question, docs)

    # TODO: GCP에서 실제 LLM 붙일 때 여기 확장
    # if backend == "openai":
    #     ...
    # if backend == "hf":
    #     ...

    raise ValueError(f"Unknown GENERATOR_BACKEND: {backend}")
