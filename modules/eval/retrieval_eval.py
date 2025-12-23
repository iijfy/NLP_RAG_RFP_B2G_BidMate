from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set

import pandas as pd

from modules.retrieval import search


def parse_gold_projects(s: str) -> Set[str]:
    if pd.isna(s):
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {x.strip() for x in s.split("|") if x.strip()}


def hit_mrr_at_k(retrieved_projects: List[str], gold_projects: Set[str], k: int):
    topk = retrieved_projects[:k]

    hit = 1.0 if any(p in gold_projects for p in topk) else 0.0

    mrr = 0.0
    for rank, p in enumerate(topk, start=1):
        if p in gold_projects:
            mrr = 1.0 / rank
            break

    return hit, mrr


def main():
    eval_path = os.getenv("EVAL_PATH", "data/eval_queries.csv")
    k = int(os.getenv("EVAL_TOP_K", "5"))
    out_path = os.getenv("EVAL_OUT", "outputs/eval_retrieval_results.csv")

    df = pd.read_csv(eval_path)
    required = {"query_id", "query_text", "gold_project_ids"}
    if not required.issubset(df.columns):
        raise ValueError(f"eval csv 컬럼이 부족합니다. 필요: {sorted(required)}")

    rows = []
    hits, mrrs = [], []

    for i, row in df.iterrows():
        qid = row["query_id"]
        query = str(row["query_text"])
        gold = parse_gold_projects(row["gold_project_ids"])

        docs = search(query, k=k)

        # ✅ project_id는 metadata['source']로 저장해둔 상태
        retrieved_projects = []
        for d in docs:
            p = d.metadata.get("source")
            if p:
                retrieved_projects.append(str(p))

        hit, mrr = hit_mrr_at_k(retrieved_projects, gold, k=k)
        hits.append(hit)
        mrrs.append(mrr)

        rows.append(
            {
                "query_id": qid,
                "query_text": query,
                "gold_project_ids": "|".join(sorted(gold)),
                f"retrieved_projects_top{k}": "|".join(retrieved_projects[:k]),
                f"hit@{k}": hit,
                f"mrr@{k}": mrr,
            }
        )

        print(f"[{i+1}/{len(df)}] id={qid} hit={hit:.0f} mrr={mrr:.3f}")

    out_df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n===== SUMMARY =====")
    print(f"count={len(df)}")
    print(f"Hit@{k}={sum(hits)/len(hits):.4f}")
    print(f"MRR@{k}={sum(mrrs)/len(mrrs):.4f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
