# modules/eval/mixed_eval.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Set, Dict, Any

import pandas as pd

from modules.retrieval import search
from modules.eval.judge_eval import judge

# ✅ RAG 답변까지 같이 평가하려면 이 함수가 있어야 함
from modules.rag import answer_query


def _parse_gold_projects(s: str) -> Set[str]:
    if pd.isna(s):
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {x.strip() for x in s.split("|") if x.strip()}


def _hit_mrr_at_k(retrieved_projects: List[str], gold_projects: Set[str], k: int):
    topk = retrieved_projects[:k]

    hit = 1.0 if any(p in gold_projects for p in topk) else 0.0

    mrr = 0.0
    first_rank = -1
    for rank, p in enumerate(topk, start=1):
        if p in gold_projects:
            mrr = 1.0 / rank
            first_rank = rank
            break

    return hit, mrr, first_rank


def main():
    eval_path = os.getenv("EVAL_PATH", "data/eval_queries.csv")
    top_k = int(os.getenv("EVAL_TOP_K", "5"))
    rag_k = int(os.getenv("RAG_TOP_K", str(top_k)))

    out_csv = os.getenv("EVAL_OUT", "outputs/eval_mixed_results.csv")
    out_jsonl = os.getenv("EVAL_JUDGMENTS_OUT", "outputs/eval_mixed_judgments.jsonl")

    df = pd.read_csv(eval_path)

    rows: List[Dict[str, Any]] = []
    judgments_fp = Path(out_jsonl)
    judgments_fp.parent.mkdir(parents=True, exist_ok=True)

    hits, mrrs = [], []
    accs, comps, profs = [], [], []

    with judgments_fp.open("w", encoding="utf-8") as fjsonl:
        for i, row in df.iterrows():
            qid = int(row["query_id"])
            query = str(row["query_text"])
            gold = _parse_gold_projects(row["gold_project_ids"])

            # 1) Retrieval 평가용 검색
            docs = search(query, k=top_k)
            retrieved_projects = [str(d.metadata.get("source")) for d in docs if d.metadata.get("source")]

            hit, mrr, first_rank = _hit_mrr_at_k(retrieved_projects, gold, k=top_k)
            hits.append(hit)
            mrrs.append(mrr)

            # 2) RAG 답변 생성(더미 가능) + G-Eval 판정
            answer, used_docs = answer_query(query, k=rag_k, docs=docs[:rag_k])
            j = judge(query, answer)

            acc = int(j.get("accuracy", 0))
            comp = int(j.get("completeness", 0))
            prof = int(j.get("professionalism", 0))

            accs.append(acc); comps.append(comp); profs.append(prof)

            # jsonl 저장(원문 로그)
            fjsonl.write(json.dumps({
                "query_id": qid,
                "query_text": query,
                "answer": answer,
                "judge": j,
            }, ensure_ascii=False) + "\n")

            rows.append({
                "query_id": qid,
                "query_text": query,
                "gold_project_ids": "|".join(sorted(gold)),
                f"retrieved_projects_top{top_k}": "|".join(retrieved_projects[:top_k]),
                f"hit@{top_k}": hit,
                f"mrr@{top_k}": mrr,
                "first_gold_rank": first_rank,

                # ✅ G-Eval 점수도 같은 row에 합침
                "g_eval_accuracy": acc,
                "g_eval_completeness": comp,
                "g_eval_professionalism": prof,
                "g_eval_rationale": str(j.get("rationale", ""))[:300],  # 너무 길면 잘라 저장
            })

            print(f"[{i+1}/{len(df)}] id={qid} hit={hit:.0f} mrr={mrr:.3f} | G(acc={acc}, comp={comp}, prof={prof})")

    out_df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n===== SUMMARY =====")
    print(f"count={len(df)}")
    print(f"Hit@{top_k}={sum(hits)/len(hits):.4f}")
    print(f"MRR@{top_k}={sum(mrrs)/len(mrrs):.4f}")
    print(f"G-Eval accuracy avg={sum(accs)/len(accs):.3f}")
    print(f"G-Eval completeness avg={sum(comps)/len(comps):.3f}")
    print(f"G-Eval professionalism avg={sum(profs)/len(profs):.3f}")
    print(f"saved -> {out_csv}")
    print(f"saved -> {out_jsonl}")


if __name__ == "__main__":
    main()
