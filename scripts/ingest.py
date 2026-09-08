import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PDF_DIR = Path("data/raw/commerce_pdfs")

TABLE_NAME = "document_chunks_1500"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def get_connection():
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

    register_vector(conn)

    return conn


def load_documents():
    all_documents = []

    for pdf_path in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_name"] = pdf_path.name
            doc.metadata["page_no"] = (
                doc.metadata["page"] + 1
            )

        all_documents.extend(docs)

    return all_documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    return splitter.split_documents(documents)


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            DROP TABLE IF EXISTS {TABLE_NAME};

            CREATE TABLE {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                start_index INTEGER,
                content TEXT NOT NULL,
                embedding VECTOR(1536)
            );
        """)

    conn.commit()


def embed_and_store(conn, chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(texts)

    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (
                        file_name,
                        page_no,
                        start_index,
                        content,
                        embedding
                    )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    chunk.metadata["file_name"],
                    chunk.metadata["page_no"],
                    chunk.metadata.get("start_index"),
                    chunk.page_content,
                    vector
                )
            )

    conn.commit()


def main():
    documents = load_documents()
    chunks = create_chunks(documents)

    print("페이지 수:", len(documents))
    print("Chunk 수:", len(chunks))

    with get_connection() as conn:
        create_table(conn)
        embed_and_store(conn, chunks)

    print("인덱싱 완료")


if __name__ == "__main__":
    main()