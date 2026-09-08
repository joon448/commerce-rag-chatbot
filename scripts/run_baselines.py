import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from app.rag import generate_rag_answer


load_dotenv()

QA_PATH = "data/processed/commerce_qa_available.csv"
PURE_PATH = "data/results/pure_llm_total.csv"
PROMPT_PATH = "data/results/prompt_v1_total.csv"
RAG_PATH = "data/results/rag_chunk1500_total.csv"


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


PROMPT = """
당신은 질문에 정확하고 간결하게 답변하는 AI 어시스턴트입니다.

- 확실하지 않은 내용은 추측하지 마세요.
- 질문에 필요한 핵심 내용만 답변하세요.
- 답을 알 수 없는 경우 "알 수 없습니다"라고 답변하세요.
"""


def run_pure(qa):
    answers = []
    input_tokens = []
    output_tokens = []
    total_tokens = []

    for _, row in qa.iterrows():
        response = client.responses.create(
            model="gpt-5-mini",
            input=row["question"]
        )

        answers.append(response.output_text)
        input_tokens.append(response.usage.input_tokens)
        output_tokens.append(response.usage.output_tokens)
        total_tokens.append(response.usage.total_tokens)

    result = qa.copy()

    result["pure_llm_answer"] = answers
    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    result["total_tokens"] = total_tokens

    result.to_csv(
        PURE_PATH,
        index=False
    )


def run_prompt(qa):
    answers = []
    input_tokens = []
    output_tokens = []
    total_tokens = []

    for _, row in qa.iterrows():
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": row["question"]
                }
            ]
        )

        answers.append(response.output_text)
        input_tokens.append(response.usage.input_tokens)
        output_tokens.append(response.usage.output_tokens)
        total_tokens.append(response.usage.total_tokens)

    result = qa.copy()

    result["prompt_v1_answer"] = answers
    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    result["total_tokens"] = total_tokens

    result.to_csv(
        PROMPT_PATH,
        index=False
    )
    
def run_rag(qa):
    rag_answers = []

    input_tokens = []
    output_tokens = []
    total_tokens = []

    retrieved_sources = []
    retrieved_distances = []

    hit_ranks = []
    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []

    for _, row in qa.iterrows():
        result = generate_rag_answer(
            row["question"],
            top_k=5
        )

        docs = result["retrieved_docs"]

        sources = [
            f"{doc[1]} / p.{doc[2]}"
            for doc in docs
        ]

        distances = [
            f"{doc[4]:.4f}"
            for doc in docs
        ]

        hit_rank = None

        for rank, doc in enumerate(
            docs,
            start=1
        ):
            _, file_name, page_no, _, _ = doc

            if (
                file_name == row["target_file_name"]
                and page_no == row["target_page_no"]
            ):
                hit_rank = rank
                break

        rag_answers.append(result["answer"])

        input_tokens.append(
            result["input_tokens"]
        )
        output_tokens.append(
            result["output_tokens"]
        )
        total_tokens.append(
            result["total_tokens"]
        )

        retrieved_sources.append(
            " | ".join(sources)
        )
        retrieved_distances.append(
            " | ".join(distances)
        )

        hit_ranks.append(hit_rank)

        hit_at_1.append(
            hit_rank is not None
            and hit_rank <= 1
        )
        hit_at_3.append(
            hit_rank is not None
            and hit_rank <= 3
        )
        hit_at_5.append(
            hit_rank is not None
            and hit_rank <= 5
        )

    result = qa.copy()

    result["rag_answer"] = rag_answers

    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    result["total_tokens"] = total_tokens

    result["retrieved_sources"] = retrieved_sources
    result["retrieved_distances"] = retrieved_distances

    result["hit_rank"] = hit_ranks
    result["hit_at_1"] = hit_at_1
    result["hit_at_3"] = hit_at_3
    result["hit_at_5"] = hit_at_5

    print(
        f"Hit@1: {result['hit_at_1'].mean():.3f}"
    )
    print(
        f"Hit@3: {result['hit_at_3'].mean():.3f}"
    )
    print(
        f"Hit@5: {result['hit_at_5'].mean():.3f}"
    )

    result.to_csv(
        RAG_PATH,
        index=False
    )
    
def main():
    qa = pd.read_csv(QA_PATH)

    run_pure(qa)
    run_prompt(qa)
    run_rag(qa)


if __name__ == "__main__":
    main()