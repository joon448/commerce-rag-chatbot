# Commerce RAG Chatbot

커머스 도메인 문서를 기반으로 질문에 답변하는 RAG 챗봇입니다.

Pure LLM, Prompt Engineering, RAG 방식을 동일한 QA 데이터셋에서 비교하고,
검색 성능과 최종 답변 품질을 평가하여 RAG 적용 효과와 한계를 분석했습니다.

---

## 1. Overview

LLM은 일반 지식을 기반으로 자연스러운 답변을 생성할 수 있지만, 특정 문서의 수치·정책·용어를 정확히 반영하지 못할 수 있습니다.

본 프로젝트에서는 다음 세 방식을 비교했습니다.

```text
Pure LLM
→ Prompt Engineering
→ Document-based RAG
```

주요 실험 항목은 다음과 같습니다.

* Prompt Engineering만으로 답변 정확도를 개선할 수 있는지
* RAG 적용 시 지정된 문서와의 정답 일치도가 향상되는지
* Chunk 설정에 따라 Retrieval 성능이 어떻게 달라지는지
* Retrieval 실패가 최종 답변에 어떤 영향을 주는지

---

## 2. Tech Stack

* **Backend**: Python, FastAPI, Uvicorn
* **LLM / RAG**: GPT-5-mini, text-embedding-3-small, LangChain
* **Parsing / Chunking**: PyPDFLoader, RecursiveCharacterTextSplitter
* **Database**: PostgreSQL, pgvector
* **Infrastructure**: Docker, Docker Compose
* **Evaluation**: Hit@K, LLM-as-a-Judge, Embedding Cosine Similarity

---

## 3. Architecture

```text
Client
  ↓ POST /ask
FastAPI
  ↓
Question Embedding
  ↓
PostgreSQL + pgvector
  ↓
Top-K Retrieval
  ↓
Context Construction
  ↓
GPT-5-mini
  ↓
Answer + Sources
```

---

## 4. Dataset

`allganize/RAG-Evaluation-Dataset-KO` 문서 기반 QA 데이터셋에서 커머스 도메인 데이터를 사용했으며, 실제 문서 다운로드가 가능한 항목만 필터링하여 최종 **35개 QA**를 평가했습니다.

각 QA는 다음 정보를 포함합니다.

```text
question
target_answer
target_file_name
target_page_no
context_type
```


---

## 5. Key Results

### 1) Retrieval Evaluation

### Chunk Size Experiment

| Chunk Size | Overlap |   Hit@1 |   Hit@3 |   Hit@5 |
| ---------: | ------: | ------: | ------: | ------: |
|        700 |     150 |     20% |     60% |     60% |
|       1000 |     200 |      0% |     60% |     80% |
|   **1500** | **300** | **20%** | **80%** | **80%** |
|       2000 |     400 |     20% |     80% |     80% |
|       2500 |     500 |     20% |     80% |     80% |

1500/300 설정에서 Hit@3가 개선되었고, 이후 Chunk Size를 늘려도 추가 향상이 없었습니다.

따라서 검색 성능과 Context 길이의 균형을 고려해 **1500 / 300**을 최종 설정으로 선택했습니다.

### Full Dataset

| Hit@1 | Hit@3 | Hit@5 |
| ----: | ----: | ----: |
| 45.7% | 74.3% | 80.0% |


---

### 2) Answer Evaluation

최종 답변은 세 가지 지표로 평가했습니다.

* **Correctness**: 핵심 답변 일치 여부, 0/1
* **LLM Similarity**: Target Answer와 생성 답변의 의미적 유사도, 0~5
* **Embedding Similarity**: 두 답변의 cosine similarity

### Results

| Method   | Correctness | LLM Similarity | Embedding Similarity | Avg Input Tokens | Avg Output Tokens |
| -------- | ----------: | -------------: | -------------------: | ---------------: | ----------------: |
| Pure LLM |       0.429 |          3.057 |                0.530 |             42.1 |            1348.4 |
| Prompt   |       0.229 |          2.057 |                0.491 |            114.1 |             845.0 |
| **RAG**  |   **0.800** |      **4.143** |            **0.702** |           2642.0 |         **614.8** |

RAG 적용 후 Correctness가 **42.9% → 80.0%**로 향상되었습니다.

Prompt Engineering은 불확실한 답변을 억제하는 데는 도움이 되었지만, 모델에 없는 도메인 지식을 추가하지 못해 Target Answer와의 일치도는 개선되지 않았습니다.

RAG는 외부 문서를 Context로 제공함으로써 특정 문서의 수치·용어·관계를 보다 정확히 반영했습니다.

> Token 수치는 LLM generation 기준이며 embedding API token은 포함하지 않았습니다.


---

## 6. RAG Pipeline

PDF 문서를 `PyPDFLoader`로 로드한 후 `RecursiveCharacterTextSplitter`로 분할했습니다.

최종 Chunk 설정:

```text
Chunk Size: 1500
Chunk Overlap: 300
```

각 Chunk는 `text-embedding-3-small`로 embedding한 뒤 PostgreSQL의 `VECTOR(1536)` 컬럼에 저장했습니다.

질문 embedding과 document embedding 간 cosine distance를 기준으로 Top-5 Chunk를 검색하고, 해당 문서를 Context로 GPT-5-mini에 전달합니다.

---


## 7. Failure Case

일부 표 기반 PDF에서는 parsing 과정에서 표 구조가 선형 텍스트로 변환되며 핵심 관계가 약해지는 문제가 있었습니다.

```text
PDF Table
→ Text Parsing
→ Table Structure Loss
→ Retrieval Quality Degradation
```

Chunk Size를 조정해도 개선되지 않은 사례가 있어, Retrieval 성능이 Chunk 설정뿐 아니라 **문서 구조 보존과 parsing 품질에도 영향을 받는다**는 점을 확인했습니다.

---

## 8. API

### Request

```http
POST /ask
```

```json
{
  "question": "2020년 글로벌 이커머스 시장 규모는 얼마인가요?"
}
```

### Response

```json
{
  "answer": "2020년 글로벌 이커머스 시장 규모는 약 4.2조 달러입니다.",
  "sources": [
    {
      "file_name": "example.pdf",
      "page_no": 5,
      "distance": 0.32
    }
  ]
}
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## 9. Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── retrieval.py
│   └── database.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── ingest.py
│   ├── run_baselines.py
│   └── evaluate.py
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 10. Run

```bash
pip install -r requirements.txt
docker compose up -d
python scripts/ingest.py
uvicorn app.main:app --reload
```

PostgreSQL에서 pgvector extension을 활성화해야 합니다.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 11. Limitations

* PDF 표 구조 손실에 따른 Retrieval 성능 저하
* 현재 dense embedding 기반 검색만 사용
* 평가 데이터 35 QA로 제한

향후 Hybrid Search, Re-ranking, Metadata Filtering, Table-aware Parsing 등을 적용해 개선할 수 있습니다.
