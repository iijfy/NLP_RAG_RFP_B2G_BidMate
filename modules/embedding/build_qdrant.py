from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from modules.paths import ProjectPaths
from modules.loader import load_chunks_df
from modules.embedding.embedder import get_embeddings
from modules.embedding.qdrant_store import build_qdrant_vectorstore


def main() -> None:
    paths = ProjectPaths()

    # 1) 어떤 청크를 인덱싱할지 선택
    #    - 지금은 네가 이미 만든 semantic/recursive CSV 둘 다 있으니 둘 중 하나 선택하면 됨
    mode = "recursive"  # "semantic" 으로 바꿔도 됨
    df = load_chunks_df(mode)

    # 2) Document로 변환 (Qdrant에 넣을 표준 형태)
    docs = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": row.get("doc_id"),
                    "chunk_id": row.get("chunk_id"),
                    "source": row.get("source"),
                    "mode": mode,
                },
            )
        )

    # 3) bge-m3 임베딩
    embeddings = get_embeddings()  # 나중에 "mps"로 바꿔볼 수 있음

    # 4) Qdrant 로컬 저장 경로
    qdrant_path = Path(paths.outputs_dir) / "qdrant_db"
    collection_name = f"rfp_{mode}_DUMMY"

    store = build_qdrant_vectorstore(
        documents=docs,
        embeddings=embeddings,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        recreate=True,
    )

    print("[OK] indexed:", collection_name)
    print("docs:", len(docs))
    print("qdrant_path:", qdrant_path)


if __name__ == "__main__":
    main()
