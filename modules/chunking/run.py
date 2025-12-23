from __future__ import annotations

from modules.loader import load_fulltext_df
from modules.chunking import chunk_recursive
from modules.paths import ProjectPaths
from modules.utils.io import write_csv


def main() -> None:
    paths = ProjectPaths()

    df = load_fulltext_df()
    chunks = chunk_recursive(df)

    # 저장 위치: data/data_list_chunks_recursive.csv 로 덮어쓰기
    write_csv(chunks, paths.csv_chunks_recursive, index=False)

    print(f"[OK] saved: {paths.csv_chunks_recursive}")
    print("shape:", chunks.shape)


if __name__ == "__main__":
    main()
