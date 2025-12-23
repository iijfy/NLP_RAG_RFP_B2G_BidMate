# modules/eval/run_eval.py
from __future__ import annotations

import os
import pandas as pd
from typing import Set, List

from modules.retrieval import search
from modules.eval.metrics import hit_and_mrr_at_k


def _parse_gold(value: str) -> Set[str]:
    """
    gold doc id가
    - "20241001798" 처럼 하나일 수도 있고
    - "20241001798|20241002912" 처럼 여러개일 수도 있어서 파싱
    """
    if pd.isna(value):
        return set()
    s = str(value).strip()
    if not s:
        return set()
    # 구분자 후보: | , ; 공백
    for sep in ["|", ",", ";"]:
        if sep in s:
            return {x.strip() for x in s.split(sep) if x.strip()}
    return {s}


def main():
    # 환경변수로 eval 파일 지정 가능하게
    eval_path = os.getenv("EVAL_PATH", "data/eval_questions.csv")
    k = int(os.getenv("EVAL_TOP_K", "5"))

    df = pd.read_csv(eval_path)

    # 최소 컬럼 가정:
    # question: 질문
    # gold_doc_id: 정답 doc_id (여러개면 |로 연결)
    if "question" not in df.columns or "gold_doc_id" not in df.columns:
        raise ValueError("eval csv에는 최소한 'question', 'gold_doc_id' 컬럼이 필요합니다.")

    hits = []
    mrrs = []

    for i, row in df.iterrows():
        q = str(row["question"])
        gold = _parse_gold(row["gold_doc_id"])

        docs = search(q, k=k)  # retriever 호출(환경변수로 컬렉션 바뀜)
        retrieved_doc_ids: List[str] = [str(d.metadata.get("doc_id")) for d in docs if d.metadata.get("doc_id")]

        res = hit_and_mrr_at_k(retrieved_doc_ids, gold, k=k)
        hits.append(res.hit)
        mrrs.append(res.mrr)

        # 진행 로그 (원하면 끌 수 있음)
        print(f"[{i+1}/{len(df)}] hit={res.hit:.0f} mrr={res.mrr:.3f} q={q[:30]}...")

    print("\n===== EVAL SUMMARY =====")
    print(f"count = {len(df)}")
    print(f"Hit@{k} = {sum(hits)/len(hits):.4f}")
    print(f"MRR@{k} = {sum(mrrs)/len(mrrs):.4f}")


if __name__ == "__main__":
    main()
