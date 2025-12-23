from __future__ import annotations

import os
from typing import List
from langchain_core.documents import Document


def generate_answer(query: str, docs: List[Document]) -> str:
    """
    답변 생성기 공통 인터페이스.

    - 로컬(VSCode)에서는 GENERATOR_BACKEND=dummy 로 실행 확인만 한다.
    - GCP에서는 openai 등으로 바꿔 끼울 수 있게 구조만 잡아둔다.
    """
    backend = os.getenv("GENERATOR_BACKEND", "dummy").lower()

    if backend == "dummy":
        # ✅ 로컬 실행 확인용: docs 메타데이터만 요약해주고 끝
        topk_lines = []
        for i, d in enumerate(docs[:3], start=1):
            meta = d.metadata or {}
            doc_id = meta.get("doc_id", "")
            chunk_id = meta.get("chunk_id", "")
            source = meta.get("source", "")
            preview = (d.page_content or "").replace("\n", " ")[:180]
            topk_lines.append(f"- [{i}] doc_id={doc_id}, chunk_id={chunk_id}, source={source} :: {preview}...")

        return (
            "※ [DUMMY ANSWER] (로컬 실행 확인용)\n"
            f"질문: {query}\n\n"
            "참고한 문서 Top-k 요약:\n"
            + "\n".join(topk_lines)
            + "\n\n"
            "최종 답변(더미): 위 문서들을 근거로 요구사항/조건/유지보수/네트워크 등 항목이 포함됩니다."
        )

    # 나중에 GCP에서 진짜 LLM 붙일 때 확장
    raise ValueError(f"Unknown GENERATOR_BACKEND={backend}")
