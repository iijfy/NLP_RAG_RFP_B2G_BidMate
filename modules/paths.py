from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True) #실수로 settings 값을 코드 중간에서 바꾸는 걸 막아줌(안전장치)
class ProjectPaths:

    root: Path = PROJECT_ROOT

    # 데이터 폴더
    data_dir: Path = PROJECT_ROOT / "data"
    data_original_dir: Path = data_dir / "data_original"

    # CSV
    csv_base: Path = data_dir / "data_list_base.csv"
    csv_chunks_recursive: Path = data_dir / "data_list_chunks_recursive.csv"
    csv_chunks_semantic: Path = data_dir / "data_list_chunks_semantic.csv"
    csv_fulltext: Path = data_dir / "data_list_fulltext.csv"

    # 출력 폴더
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    logs_dir: Path = outputs_dir / "logs"
