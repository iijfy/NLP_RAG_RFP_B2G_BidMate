from __future__ import annotations

from typing import Optional
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_recursive(
    df_fulltext: pd.DataFrame,
    *,
    text_col: str = "full_text",      # ✅ 너 DF 컬럼명
    doc_id_col: str = "공고 번호",     # ✅ 너 DF 컬럼명
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    source_col: Optional[str] = "project_id",
) -> pd.DataFrame:
    """
    노트북 셀에서 하던 "Recursive 청킹"을 함수로 뽑아낸 버전.

    반환 DF 컬럼:
      - doc_id: 문서 id(공고 번호)
      - chunk_id: 문서 내 청크 번호
      - text: 청크 텍스트
      - source: 출처(있으면 project_id 같은 값)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    rows = []
    for _, row in df_fulltext.iterrows():
        doc_id = str(row[doc_id_col])
        text = row[text_col]

        if not isinstance(text, str) or not text.strip():
            continue

        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "text": ch,
                    "source": row[source_col] if source_col and source_col in df_fulltext.columns else None,
                }
            )

    return pd.DataFrame(rows)
