# 모듈화하면 chunk dataframe 컬럼 규격을 강제로 고정시키는 게 유지보수에 좋다고 함

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkColumns:
    """
    chunk dataframe의 컬럼 이름을 한 곳에서 관리.
    나중에 retrieval/eval에서 '어떤 컬럼을 참조해야 하는지'가 절대 안 흔들림.
    """
    doc_id: str = "doc_id"          # 문서 식별자(공고번호 등)
    chunk_id: str = "chunk_id"      # 문서 내 청크 인덱스
    text: str = "text"              # 청크 본문
    source: str = "source"          # pdf 경로 or 문서 출처
    page: str = "page"              # (있으면) 페이지 정보
    meta: str = "meta"              # (있으면) 기타 메타(JSON string 등)
