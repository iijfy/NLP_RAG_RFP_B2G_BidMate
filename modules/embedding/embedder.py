from __future__ import annotations

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings


def _auto_device() -> str:
    """
    torch가 있으면 cuda/mps 사용 가능 여부를 보고,
    없거나 문제가 있으면 cpu로 폴백.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        # Mac Apple Silicon(M1/M2/M3)에서만 보통 True
        if getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"

    except Exception:
        # torch import가 깨지거나(예: numpy 충돌) 기타 문제면
        # 일단 cpu로 가서 파이프라인이 멈추지 않게 함
        pass

    return "cpu"


def get_bge_m3_embeddings(device: Optional[str] = None, *, verbose: bool = True) -> HuggingFaceEmbeddings:
    """
    - device=None이면 자동 선택(cuda → mps → cpu)
    - device="cpu"/"mps"/"cuda"로 강제 지정 가능
    """
    chosen = device or _auto_device()

    if verbose:
        print(f"[Embeddings] device = {chosen}")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": chosen},
        encode_kwargs={"normalize_embeddings": True},
    )
