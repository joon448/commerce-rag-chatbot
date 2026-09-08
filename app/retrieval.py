import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.database import get_connection


load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

TABLE_NAME = "document_chunks_1500"


def retrieve(question: str, top_k: int = 5):
    query_vector = embeddings.embed_query(question)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    file_name,
                    page_no,
                    content,
                    embedding <=> %s::vector AS distance
                FROM {TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k)
            )

            return cur.fetchall()