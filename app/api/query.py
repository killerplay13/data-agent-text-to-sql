from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService
from app.services.execution_service import ExecutionService
from app.services.answer_service import AnswerService


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_data(request: QueryRequest):
    try:
        retrieval_service = RetrievalService()
        sql_service = SQLGenerationService()
        execution_service = ExecutionService()
        answer_service = AnswerService()

        retrieval_result = retrieval_service.retrieve(request.query)
        generated_sql = sql_service.generate_sql(request.query, retrieval_result)
        query_result = execution_service.execute_query(generated_sql)
        answer = answer_service.generate_answer(request.query, query_result)

        return QueryResponse(
            answer=answer,
            generated_sql=generated_sql,
            query_result=query_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))