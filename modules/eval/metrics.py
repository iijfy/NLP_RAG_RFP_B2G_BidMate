# modules/eval/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict


@dataclass
class EvalResult:
    hit: float
    mrr: float


def hit_and_mrr_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> EvalResult:
    """
    retrieved_doc_ids: retriever 결과에서 doc_id만 뽑은 리스트 (순서 중요)
    gold_doc_ids: 정답 doc_id 집합 (정답이 1개면 set에 1개)
    """
    topk = retrieved_doc_ids[:k]

    # Hit@k
    hit = 1.0 if any(d in gold_doc_ids for d in topk) else 0.0

    # MRR@k
    mrr = 0.0
    for rank, d in enumerate(topk, start=1):
        if d in gold_doc_ids:
            mrr = 1.0 / rank
            break

    return EvalResult(hit=hit, mrr=mrr)
