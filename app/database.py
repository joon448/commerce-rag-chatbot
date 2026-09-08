import os
import psycopg

from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv()


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