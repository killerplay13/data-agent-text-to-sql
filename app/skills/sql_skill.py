import re

from app.services.sql_generation_service import SQLGenerationService
from app.skills.base_skill import BaseSkill


KNOWN_TABLES = {"customers", "deposits", "branches", "relationship_managers"}
MALFORMED_SQL_PATTERN = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|ON|GROUP|ORDER|BY|HAVING|LIMIT)(?=[A-Za-z_])",
    re.IGNORECASE,
)


class SQLSkill(BaseSkill):
    name = "sql_generation"

    def __init__(self):
        self.service = SQLGenerationService()

    def execute(self, input: dict) -> dict:
        input["generated_sql"] = self.service.generate_sql(
            input["user_query"],
            input["retrieval_result"],
        )

        if not self._is_valid_sql(input["generated_sql"], input["user_query"]):
            print("Generated SQL failed validation; falling back to retrieved SQL template.")
            input["generated_sql"] = self._fallback_sql(input["retrieval_result"])

        return input

    def _is_valid_sql(self, sql: str, user_query: str) -> bool:
        if not sql or not isinstance(sql, str):
            return False

        sql_clean = sql.strip()
        sql_lower = sql_clean.lower()
        user_query_lower = user_query.lower()

        if not sql_lower.startswith("select"):
            return False
        if MALFORMED_SQL_PATTERN.search(sql_clean):
            return False
        if not any(re.search(rf"\b{table}\b", sql_lower) for table in KNOWN_TABLES):
            return False
        if self._mentions_relationship_manager(user_query_lower) and "customers" not in sql_lower:
            return False
        if "branch" in user_query_lower and "branches" not in sql_lower:
            return False

        return True

    def _mentions_relationship_manager(self, user_query: str) -> bool:
        return (
            "relationship manager" in user_query
            or "managed by" in user_query
            or "manager" in user_query
        )

    def _fallback_sql(self, retrieval_result: dict) -> str:
        return retrieval_result["sql_templates"][0]["sql"]
