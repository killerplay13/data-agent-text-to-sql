from pydantic import BaseModel
from typing import Any


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    generated_sql: str
    query_result: list[dict[str, Any]]