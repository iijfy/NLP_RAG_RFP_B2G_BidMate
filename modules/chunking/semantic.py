from __future__ import annotations

import pandas as pd
from typing import Optional

from modules.loader import load_chunks_df


def load_semantic_chunks() -> pd.DataFrame:
    """
    너가 이미 만들어둔 semantic chunk 결과 CSV를 불러오는 함수.
    (semantic 알고리즘 재구현은 다음 단계에서 해도 됨)
    """
    return load_chunks_df("semantic")
