from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import generate_rag_answer


app = FastAPI(
    title="Commerce RAG Chatbot",
    description="커머스 도메인 문서 기반 RAG 챗봇",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="사용자 질문"
    )


class Source(BaseModel):
    file_name: str
    page_no: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {
        "message": "Commerce RAG Chatbot API"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="질문을 입력해주세요."
            )

        return generate_rag_answer(
            question=request.question
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="답변 생성 중 오류가 발생했습니다."
        )