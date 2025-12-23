from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import ast

import pandas as pd

from modules.paths import ProjectPaths
from modules.utils.io import read_csv


ChunkMode = Literal["recursive", "semantic"]


def _parse_list_cell(cell: object) -> list[str]:
    """
    CSV의 pdf_list 컬럼은 종종 아래처럼 '문자열'로 저장돼요.
    "['/content/drive/.../a.pdf', '/content/drive/.../b.pdf']"

    그래서 로딩 후에 실제 리스트(list[str])로 바꿔주는 함수가 필요합니다.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    # 이미 list면 그대로
    if isinstance(cell, list):
        return [str(x) for x in cell]

    # 문자열이면 안전하게 파싱
    if isinstance(cell, str):
        cell = cell.strip()
        if cell == "":
            return []
        try:
            parsed = ast.literal_eval(cell)  # eval보다 안전한 파서
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            # 형식이 깨져 있으면 단일 값으로 취급
            return [cell]

    # 그 외 타입은 문자열 1개로 처리
    return [str(cell)]


def _build_pdf_index(data_original_dir: Path) -> dict[str, list[Path]]:
    """
    data_original 아래의 모든 PDF를 스캔해서
    {파일명: [경로들]} 형태로 인덱스를 만듭니다.

    왜 필요?
    - CSV의 pdf_list에는 코랩 경로가 들어있어서 로컬에서 못 열어요.
    - 대신 '파일명(basename)'로 로컬 data_original에서 찾아 매칭시키는 방식이 가장 튼튼합니다.
    """
    index: dict[str, list[Path]] = {}

    pdf_paths = list(data_original_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"data_original에서 PDF를 찾지 못했습니다: {data_original_dir}"
        )

    for p in pdf_paths:
        index.setdefault(p.name, []).append(p)

    return index


def _resolve_pdf_paths(
    pdf_list: list[str],
    pdf_index: dict[str, list[Path]],
    *,
    strict: bool = False,
) -> list[str]:
    """
    코랩 경로 리스트를 "로컬 파일 경로 리스트"로 변환합니다.

    매칭 규칙:
    - 각 항목에서 파일명만 뽑아서(data.pdf)
    - data_original 인덱스에서 같은 파일명을 찾아 경로로 치환

    strict=False:
      못 찾으면 그 항목은 버리고 진행 (실험/디버깅에 편함)
    strict=True:
      못 찾으면 예외 발생 (파이프라인 품질 관리에 좋음)
    """
    resolved: list[str] = []

    for raw in pdf_list:
        name = Path(raw).name  # "/content/.../abc.pdf" -> "abc.pdf"
        candidates = pdf_index.get(name, [])

        if not candidates:
            if strict:
                raise FileNotFoundError(f"로컬에서 PDF를 찾지 못함: {name} (원본: {raw})")
            continue

        # 같은 파일명이 여러 개면 일단 첫 번째 사용
        # (나중에 중복이 많다면 공고번호/폴더규칙 기반으로 더 정교하게 개선 가능)
        resolved.append(str(candidates[0]))

    return resolved


def _fix_pdf_list_column(
    df: pd.DataFrame,
    paths: ProjectPaths,
    *,
    pdf_list_col: str = "pdf_list",
    strict: bool = False,
) -> pd.DataFrame:
    """
    df['pdf_list']를 로컬 경로로 복구한 df를 반환합니다.
    """
    if pdf_list_col not in df.columns:
        # 어떤 CSV에는 pdf_list가 없을 수도 있어서 조용히 통과
        return df

    pdf_index = _build_pdf_index(paths.data_original_dir)

    fixed = df.copy()
    fixed[pdf_list_col] = fixed[pdf_list_col].apply(_parse_list_cell)
    fixed[pdf_list_col] = fixed[pdf_list_col].apply(
        lambda lst: _resolve_pdf_paths(lst, pdf_index, strict=strict)
    )
    return fixed



# 공개 API: 여기부터가 "loader 모듈"의 핵심
def load_base_df(
    paths: Optional[ProjectPaths] = None,
    *,
    fix_pdf_list: bool = True,
    strict_pdf_match: bool = False,
) -> pd.DataFrame:
    """
    data_list_base.csv 로드.

    fix_pdf_list=True면:
      - pdf_list 컬럼을 로컬 경로로 자동 복구
    """
    paths = paths or ProjectPaths()
    df = read_csv(paths.csv_base)

    if fix_pdf_list:
        df = _fix_pdf_list_column(df, paths, strict=strict_pdf_match)

    return df


def load_fulltext_df(
    paths: Optional[ProjectPaths] = None,
) -> pd.DataFrame:
    """
    data_list_fulltext.csv 로드.
    - 보통 fulltext는 이미 텍스트로 정리된 상태라 pdf_list 복구가 필요 없는 경우가 많아요.
    """
    paths = paths or ProjectPaths()
    return read_csv(paths.csv_fulltext)


def load_chunks_df(
    mode: ChunkMode,
    paths: Optional[ProjectPaths] = None,
) -> pd.DataFrame:
    """
    청킹 결과 CSV 로드.

    mode:
      - "recursive"  -> data_list_chunks_recursive.csv
      - "semantic"   -> data_list_chunks_semantic.csv
    """
    paths = paths or ProjectPaths()

    if mode == "recursive":
        return read_csv(paths.csv_chunks_recursive)
    if mode == "semantic":
        return read_csv(paths.csv_chunks_semantic)

    raise ValueError(f"mode는 'recursive' 또는 'semantic' 이어야 합니다. 현재: {mode}")
