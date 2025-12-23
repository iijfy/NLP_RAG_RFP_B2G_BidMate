from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    
    path.mkdir(parents=True, exist_ok=True)


def read_csv(csv_path: Path) -> pd.DataFrame:
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")
    return pd.read_csv(csv_path)


def write_csv(df: pd.DataFrame, csv_path: Path, index: bool = False) -> None:
    
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=index)
