from pydantic import AliasChoices, BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    user_query: str = Field(
        validation_alias=AliasChoices("user_query", "query"),
        description="Natural language query from the user.",
    )


class QueryResponse(BaseModel):
    answer: str
    generated_sql: str
    query_result: list[dict[str, Any]]
