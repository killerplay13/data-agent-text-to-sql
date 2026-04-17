from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.skills.answer_skill import AnswerSkill
from app.skills.execution_skill import ExecutionSkill
from app.skills.retrieval_skill import RetrievalSkill
from app.skills.sql_skill import SQLSkill


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_data(request: QueryRequest):
    try:
        skills = [
            RetrievalSkill(),
            SQLSkill(),
            ExecutionSkill(),
            AnswerSkill(),
        ]

        context = {"user_query": request.user_query}
        for skill in skills:
            context = skill.execute(context)

        return QueryResponse(
            answer=context["answer"],
            generated_sql=context["generated_sql"],
            query_result=context["query_result"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))