import os

from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval import retrieve


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


RAG_SYSTEM_PROMPT = """
당신은 제공된 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

- 반드시 제공된 Context를 근거로 답변하세요.
- Context에 없는 내용은 추측하거나 만들어내지 마세요.
- 질문에 필요한 핵심 내용만 간결하게 답변하세요.
- Context만으로 답할 수 없는 경우 '알 수 없습니다'라고 답변하세요.
"""


def build_context(results):
    contexts = []

    for rank, row in enumerate(results, start=1):
        _, file_name, page_no, content, distance = row

        contexts.append(
            f"[Document {rank}]\n"
            f"File: {file_name}\n"
            f"Page: {page_no}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(contexts)


def generate_rag_answer(question: str, top_k: int = 5):
    results = retrieve(
        question=question,
        top_k=top_k
    )

    context = build_context(results)

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ]
    )

    sources = [
        {
            "file_name": row[1],
            "page_no": row[2],
            "distance": row[4],
        }
        for row in results
    ]

    return {
        "answer": response.output_text,
        "sources": sources,
        "retrieved_docs": results,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }