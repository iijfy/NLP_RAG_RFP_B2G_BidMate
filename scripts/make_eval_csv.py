# scripts/make_eval_csv.py
from pathlib import Path
import pandas as pd
from modules.eval.questions import EVAL_QUERIES

def main():
    rows = []
    for item in EVAL_QUERIES:
        rows.append({
            "query_id": item["id"],
            "query_text": item["query"],
            "gold_project_ids": "|".join(item["gold_project_ids"]),  # 리스트 -> 문자열
        })

    df = pd.DataFrame(rows).sort_values("query_id")

    out = Path("data") / "eval_queries.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"[OK] saved -> {out} rows={len(df)}")
    print(df.head(3))

if __name__ == "__main__":
    main()
