from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
from langchain_core.documents import Document

from modules.rag import answer_query

import re

def _query_quality_score(query: str) -> int:
    """
    더미 단계용 '질문 품질 점수(1~10)' 휴리스틱.
    - 길이, 구체성(기관/사업명/조건/숫자), 요청 형태(비교/표/요약 등)를 보고 점수 부여
    - GCP에서 LLM Judge 붙이면 이 부분을 교체하면 됨.
    """
    q = (query or "").strip()
    if not q:
        return 1

    score = 3  # 기본점수

    # 1) 너무 짧으면 패널티
    if len(q) < 8:
        score -= 2

    # 2) 구체성: 숫자/기간/Top-k/예산 등 조건이 있으면 가산
    if re.search(r"\d", q):                # 숫자 포함
        score += 1
    if re.search(r"(상위|TOP|top)\s*\d+", q):
        score += 2
    if re.search(r"(억|만원|천만원|백만원|예산|금액)", q):
        score += 2
    if re.search(r"(202\d|기간|마감|일정|까지|이후|전후)", q):
        score += 1

    # 3) 의도/요청 형태: 표/비교/요약/정리 같은 명령이 있으면 가산
    if re.search(r"(정리|요약|비교|표로|찾아|추출|분석|나열|알려줘|설명해줘)", q):
        score += 2

    # 4) 대상(기관/대학/사업명)처럼 고유명사 느낌이 있으면 가산
    if re.search(r"(대학교|대학|연구원|시청|공단|재단|부|청|위원회|시스템|포털|학사)", q):
        score += 2

    # 5) 애매한 지시어만 있으면 감점
    if re.fullmatch(r"(이거|그거|저거|이것|그것|저것)\s*(해줘|알려줘|뭐야)?", q):
        score -= 2

    # clamp
    score = max(1, min(10, score))
    return score


def _format_citations(docs: List[Document], k: int) -> str:
    lines = []
    for i, d in enumerate(docs[:k], start=1):
        meta = d.metadata or {}
        doc_id = meta.get("doc_id", "")
        chunk_id = meta.get("chunk_id", "")
        source = meta.get("source", "")
        mode = meta.get("mode", "")

        file_name = (
            meta.get("file_name")
            or meta.get("pdf_name")
            or meta.get("file")
            or ""
        )
        file_part = f" | file={file_name}" if file_name else ""

        lines.append(
            f"- [{i}] doc_id={doc_id} | chunk_id={chunk_id} | source={source} | mode={mode}{file_part}"
        )
    return "\n".join(lines).strip()



def _format_sources(docs: List[Document], max_chars: int = 350) -> str:
    """
    검색된 문서 Top-k를 사람이 보기 좋게 요약해서 보여줍니다.
    - metadata(doc_id, chunk_id, source) + 본문 앞부분만 출력
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        doc_id = meta.get("doc_id", "")
        chunk_id = meta.get("chunk_id", "")
        source = meta.get("source", "")
        mode = meta.get("mode", "")

        text = (d.page_content or "").replace("\n", " ").strip()
        preview = text[:max_chars] + ("..." if len(text) > max_chars else "")

        lines.append(
            f"[{i}] doc_id={doc_id} | chunk_id={chunk_id} | source={source} | mode={mode}\n"
            f"{preview}\n"
        )
    return "\n".join(lines).strip()


def run(query: str, rag_top_k: int) -> Tuple[str, str]:
    query = (query or "").strip()
    if not query:
        return "질문을 입력해줘.", ""

    rag_top_k = int(rag_top_k)

    # RAG 실행 (더미 generator 사용)
    answer, docs = answer_query(query, k=rag_top_k)

    # 1) 질문 품질 점수(1~10)
    q_score = _query_quality_score(query)

    # 2) 참고 문서(답변에 포함)
    citations = _format_citations(docs, k=rag_top_k)

    # 3) 답변 텍스트를 '보고서 스타일'로 감싸기
    answer_with_meta = (
        f"질문 품질 점수: {q_score}/10\n\n"
        f"참고 문서(답변 근거, Top-{rag_top_k}):\n"
        f"{citations}\n\n"
        f"---\n"
        f"{answer}"
    )

    # 기존 아래 박스(참고문서 요약)도 유지
    sources_text = _format_sources(docs[:rag_top_k])

    return answer_with_meta, sources_text



def build_demo() -> gr.Blocks:
    """
    Gradio UI 구성. (더미용)
    환경변수:
      - RAG_TOP_K: 기본 top-k
    """
    default_k = int(os.getenv("RAG_TOP_K", "3"))

    with gr.Blocks(title="RAG Demo (Dummy)") as demo:
        gr.Markdown(
            "# RAG 데모 (더미)\n"
            "- 현재는 로컬 실행 확인용 더미(generator/judge)를 씁니다.\n"
            "- 검색(Qdrant) 결과 Top-k를 보여주고, 더미 답변을 출력합니다."
        )

        with gr.Row():
            query = gr.Textbox(
                label="질문",
                placeholder="예) 차세대 학사정보시스템 구축 사업의 주요 요구사항은?",
                lines=2,
            )

        with gr.Row():
            rag_top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=default_k,
                step=1,
                label="RAG Top-k (답변에 사용할 문서 수)",
            )
            btn = gr.Button("질문하기", variant="primary")

        with gr.Row():
            out_answer = gr.Textbox(label="답변(더미)", lines=12)
        with gr.Row():
            out_sources = gr.Textbox(label="참고 문서 Top-k(요약)", lines=14)

        btn.click(
            fn=run,
            inputs=[query, rag_top_k],
            outputs=[out_answer, out_sources],
        )

    return demo
