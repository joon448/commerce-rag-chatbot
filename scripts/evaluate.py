import json
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm


load_dotenv()


# -------------------------
# 설정
# -------------------------

EVAL_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

RESULT_FILES = {
    "Pure LLM": {
        "path": "data/results/pure_llm_total.csv",
        "answer_column": "pure_llm_answer",
        "output_path": "data/results/pure_llm_evaluated.csv",
    },
    "Prompt": {
        "path": "data/results/prompt_v1_total.csv",
        "answer_column": "prompt_v1_answer",
        "output_path": "data/results/prompt_v1_evaluated.csv",
    },
    "RAG": {
        "path": "data/results/rag_chunk1500_total.csv",
        "answer_column": "rag_answer",
        "output_path": "data/results/rag_chunk1500_evaluated.csv",
    },
}


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)


# -------------------------
# LLM Judge Prompt
# -------------------------

EVAL_PROMPT = """
당신은 질의응답 결과를 평가하는 심사자입니다.

question, target_answer, generated_answer를 비교하여
다음 두 항목을 평가하세요.

1. correctness
- 질문에 대한 핵심 답변이 target_answer와 일치하면 1
- 핵심 사실, 수치, 관계가 틀리거나
  중요한 내용이 빠져 답변으로 보기 어려우면 0
- 표현 방식이 달라도 의미가 같으면 1로 평가하세요.

2. similarity_score
- target_answer와 generated_answer의 의미적 유사도를
  0~5로 평가하세요.

5: 핵심 의미와 세부 내용이 거의 동일
4: 핵심 의미는 동일하나 일부 세부 내용 차이 또는 누락
3: 핵심 내용 일부만 일치
2: 관련은 있으나 중요한 내용이 상당히 다름
1: 약간 관련만 있음
0: 의미적으로 거의 무관

질문의 맥락을 반드시 고려하세요.
"""


# -------------------------
# LLM 평가
# -------------------------

def llm_evaluate(
    question: str,
    target_answer: str,
    generated_answer: str
) -> dict:

    response = client.responses.create(
        model=EVAL_MODEL,
        instructions=EVAL_PROMPT,
        input=f"""
Question:
{question}

Target Answer:
{target_answer}

Generated Answer:
{generated_answer}
""",
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "correctness": {
                            "type": "integer",
                            "enum": [0, 1],
                        },
                        "similarity_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                        },
                    },
                    "required": [
                        "correctness",
                        "similarity_score",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(response.output_text)


# -------------------------
# Embedding Similarity
# -------------------------

def cosine_similarity(vec1, vec2):
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    return float(
        np.dot(vec1, vec2)
        / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )


def embedding_similarity(
    target_answer: str,
    generated_answer: str
) -> float:

    vectors = embeddings.embed_documents(
        [
            target_answer,
            generated_answer,
        ]
    )

    return cosine_similarity(
        vectors[0],
        vectors[1]
    )


# -------------------------
# DataFrame 평가
# -------------------------

def evaluate_dataframe(
    df: pd.DataFrame,
    answer_column: str
) -> pd.DataFrame:

    result_df = df.copy()

    correctness_list = []
    llm_similarity_list = []
    embedding_similarity_list = []

    for _, row in tqdm(
        result_df.iterrows(),
        total=len(result_df),
        desc=f"Evaluating {answer_column}"
    ):
        llm_result = llm_evaluate(
            question=row["question"],
            target_answer=row["target_answer"],
            generated_answer=row[answer_column],
        )

        emb_similarity = embedding_similarity(
            target_answer=row["target_answer"],
            generated_answer=row[answer_column],
        )

        correctness_list.append(
            llm_result["correctness"]
        )

        llm_similarity_list.append(
            llm_result["similarity_score"]
        )

        embedding_similarity_list.append(
            emb_similarity
        )

    result_df["correctness"] = correctness_list
    result_df["llm_similarity"] = llm_similarity_list
    result_df["embedding_similarity"] = (
        embedding_similarity_list
    )

    return result_df


# -------------------------
# 결과 요약
# -------------------------

def summarize_result(
    method: str,
    df: pd.DataFrame
) -> dict:

    summary = {
        "method": method,
        "correctness": df["correctness"].mean(),
        "llm_similarity": df["llm_similarity"].mean(),
        "embedding_similarity": (
            df["embedding_similarity"].mean()
        ),
    }

    # baseline/RAG CSV에 token 정보가 있는 경우
    if "input_tokens" in df.columns:
        summary["avg_input_tokens"] = (
            df["input_tokens"].mean()
        )

    if "output_tokens" in df.columns:
        summary["avg_output_tokens"] = (
            df["output_tokens"].mean()
        )

    return summary


# -------------------------
# 실행
# -------------------------

def main():
    summaries = []

    for method, config in RESULT_FILES.items():

        print(f"\n=== {method} ===")

        df = pd.read_csv(
            config["path"]
        )

        evaluated_df = evaluate_dataframe(
            df=df,
            answer_column=config["answer_column"]
        )

        evaluated_df.to_csv(
            config["output_path"],
            index=False
        )

        summary = summarize_result(
            method,
            evaluated_df
        )

        summaries.append(summary)

        print(
            f"Correctness: "
            f"{summary['correctness']:.3f}"
        )
        print(
            f"LLM Similarity: "
            f"{summary['llm_similarity']:.3f}"
        )
        print(
            f"Embedding Similarity: "
            f"{summary['embedding_similarity']:.3f}"
        )

    # -------------------------
    # 최종 비교표
    # -------------------------

    summary_df = pd.DataFrame(summaries)

    print("\n=== Final Evaluation Summary ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(
        "data/results/evaluation_summary.csv",
        index=False
    )


if __name__ == "__main__":
    main()