# modules/eval/judge_eval.py
from __future__ import annotations

import json
import os
from typing import Dict, Any


def judge_dummy(query: str, answer: str) -> Dict[str, Any]:
    # 로컬 실행 확인용 (의미 있는 평가는 GCP에서)
    return {
        "accuracy": 3,
        "completeness": 3,
        "professionalism": 3,
        "rationale": "DUMMY 판사: 로컬 스모크 테스트용 고정 점수",
    }


def judge_openai(query: str, answer: str) -> Dict[str, Any]:
    """
    PROJECT.ipynb의 G-Eval 아이디어를 .py로 옮긴 버전.
    - 3가지 기준(정확성/완전성/전문성)
    - JSON으로만 출력하게 강제 (파싱 쉬움)
    """
    from langchain_openai import ChatOpenAI

    model = os.getenv("JUDGE_MODEL", "gpt-5-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = f"""
당신은 10년 차 입찰 전문 컨설턴트입니다.
다음 AI 답변을 평가하세요.

[질문]
{query}

[답변]
{answer}

평가 기준(각 1~5점, 정수):
1) 정확성: 사실/근거 일치, 할루시네이션 없음
2) 완전성: 질문 요구사항 충족, 누락 최소
3) 전문성: 입찰 컨설팅 관점에서 실무적으로 유용(리스크/전략/확인포인트)

반드시 아래 JSON만 출력하세요 (설명 문장 금지):
{{
  "accuracy": <1~5 int>,
  "completeness": <1~5 int>,
  "professionalism": <1~5 int>,
  "rationale": "<짧게 이유>"
}}
""".strip()

    resp = llm.invoke(prompt).content.strip()

    # 혹시 앞뒤에 잡텍스트 붙으면 JSON만 추출
    start = resp.find("{")
    end = resp.rfind("}")
    if start != -1 and end != -1:
        resp = resp[start : end + 1]

    return json.loads(resp)


def judge(query: str, answer: str) -> Dict[str, Any]:
    backend = os.getenv("JUDGE_BACKEND", "dummy").lower()
    if backend == "openai":
        return judge_openai(query, answer)
    return judge_dummy(query, answer)
