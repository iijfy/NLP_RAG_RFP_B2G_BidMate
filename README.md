# 📝 NLP_RAG_RFP_B2G_BidMate

공공·기업 RFP 문서를 대상으로 핵심 요구사항/예산/제출정보를 추출·요약하는 사내 RAG 시스템입니다.  
PDF/HWP 로딩, 청킹·임베딩·리트리벌 고도화와 평가 지표 설계를 포함합니다.

---

## 📌 1. 프로젝트 목표

- RFP 문서를 검색 가능한 형태로 정리하고(Qdrant 인덱싱)
- 질의에 대해 관련 문서를 찾아(Retrieval)
- 검색 근거 기반으로 답변을 생성(RAG)
- Retrieval 성능(Hit/MRR)과 답변 품질(LLM Judge)을 함께 평가합니다.

---

## 2. 전체 파이프라인

- 데이터 로딩: CSV 기반 데이터 로드(원본 문서 경로/메타 포함)
- 청킹: Recursive 방식으로 텍스트를 일정 길이로 분할
- 임베딩: BAAI/bge-m3 임베딩 생성
- 벡터 저장: Qdrant(Local DB)에 컬렉션 저장
- 검색: Qdrant similarity search
- 답변 생성: LLM(OpenAI) 또는 더미 생성기
- 평가:
  - Retrieval: Hit@K, MRR@K
  - Answer: 정확성/완전성/전문성(1~5) LLM Judge

---

## 3. 실행 환경

- Python: 3.12
- GPU: NVIDIA L4 (GCP)
- Embeddings: BAAI/bge-m3
- Vector DB: Qdrant (local mode, outputs/qdrant_db)
- LLM/Judge: OpenAI (환경변수로 on/off)

---

## 4. 프로젝트 구조

- app.py: Gradio UI 실행
- modules/
  - loader/: CSV 로딩 및 경로 보정
  - chunking/: recursive/semantic 청킹
  - embedding/: 임베딩 생성 및 Qdrant 인덱싱
  - retrieval/: 검색(단일 entrypoint: search)
  - rag/: query -> retrieval -> answer 파이프라인
  - generator/: 답변 생성기(dummy/openai)
  - eval/: retrieval + rag answer 혼합 평가
  - ui/: gradio 앱
  - utils/: 입출력 유틸
  - paths.py: 프로젝트 경로 정의
- data/: 입력 CSV 및 평가 쿼리 CSV
- outputs/:
  - qdrant_db/: Qdrant 로컬 DB
  - eval_retrieval_results.csv: 평가 결과 CSV
  - eval_mixed_judgments.jsonl: Judge 로그(JSONL)

---

## 5. 설치

가상환경 활성화 후 requirements 설치를 권장합니다.

```bash
source ~/morgan_env/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

---

## 6. 인덱싱 (Qdrant 빌드)

임베딩 모델(bge-m3)로 청크를 임베딩한 뒤, Qdrant 컬렉션에 저장합니다.

```bash
source ~/morgan_env/bin/activate
cd ~/NLP_RAG_RFP_B2G_BidMate

export EMBEDDINGS_BACKEND="hf"
export EMBEDDING_MODEL_NAME="BAAI/bge-m3"
export EMBEDDING_DEVICE="cuda"
export QDRANT_COLLECTION="rfp_recursive"

PYTHONPATH=$(pwd) python -m modules.embedding.build_qdrant

---

## 7. 검색 스모크 테스트

```bash
source ~/morgan_env/bin/activate
cd ~/NLP_RAG_RFP_B2G_BidMate

export QDRANT_COLLECTION="rfp_recursive"

PYTHONPATH=$(pwd) python -m modules.retrieval.test_search

---

## 8. Gradio UI 실행

```bash
source ~/morgan_env/bin/activate
cd ~/NLP_RAG_RFP_B2G_BidMate

PYTHONPATH=$(pwd) python app.py

---

## 9. 평가 실행 (Retrieval + Answer 혼합 평가)

평가 입력은 data/eval_queries.csv를 사용합니다.
- Retrieval 평가: Hit@K, MRR@K
- Answer 평가: LLM Judge로 3가지 점수(정확성/완전성/전문성)

```bash
source ~/morgan_env/bin/activate
cd ~/NLP_RAG_RFP_B2G_BidMate

export EVAL_PATH="data/eval_queries.csv"
export EVAL_TOP_K="5"
export RAG_TOP_K="3"

export GENERATOR_BACKEND="openai"
export JUDGE_BACKEND="openai"
export JUDGE_MODEL="gpt-5-mini"

PYTHONPATH=$(pwd) python -m modules.eval.mixed_eval

---

10. 평가 결과

count = 20
Hit@5 = 0.9
MRR@5 = 0.8375
G-Eval accuracy avg = 4.6
G-Eval completeness avg = 3.9
G-Eval professionalism avg = 4.2

저장 파일:
outputs/eval_retrieval_results.csv
outputs/eval_mixed_judgments.jsonl

---

11. 해석 및 개선 방향

Hit@5는 0.9로 대부분의 질의에서 정답 프로젝트를 Top-5 내에 포함했습니다.
hit=0이 발생한 질의가 존재하므로(20개 중 2개) 다음 개선을 우선순위로 둡니다.

개선 후보:
- 청킹 파라미터 튜닝: chunk_size/overlap, separators, 문단/표 구조 보존
- 메타데이터 강화: 사업명/기관/연도/요구사항 키워드 등 구조화 필드 추가
- Hybrid retrieval: sparse(BM25) + dense(bge-m3) 결합
- Reranker 추가: Top-20 후보 후 rerank로 Top-5 품질 개선
- Query rewriting: 질문을 검색 친화적으로 변환해 recall 개선

---

12. 재현 체크리스트

Qdrant 컬렉션 생성 완료(rfp_recursive)
검색 테스트에서 docs가 반환되는지 확인
mixed_eval 실행 후 outputs에 csv/jsonl 생성 확인
hit=0 케이스는 eval csv와 retrieved 결과를 비교해 원인 분석 가능