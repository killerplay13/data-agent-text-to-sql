import re

from app.services.execution_service import ExecutionService
from app.skills.base_skill import BaseSkill


DEFAULT_KNOWN_TABLES = {
    "branches",
    "relationship_managers",
    "customers",
    "deposits",
}
SQL_KEYWORDS = "SELECT|FROM|WHERE|JOIN|ON|GROUP|ORDER|BY|HAVING|LIMIT"


class ExecutionSkill(BaseSkill):
    name = "execution"

    def __init__(self):
        self.service = ExecutionService()

    def execute(self, input: dict) -> dict:
        if not self._is_valid_sql(input["generated_sql"], input.get("retrieval_result", {})):
            input["generated_sql"] = self._fallback_sql(input.get("retrieval_result", {}))

        input["query_result"] = self.service.execute_query(input["generated_sql"])
        return input

    def _is_valid_sql(self, sql: str, retrieval_result: dict) -> bool:
        if not isinstance(sql, str):
            return False

        sql_clean = sql.strip()
        if not sql_clean.lower().startswith("select"):
            return False

        known_tables = self._known_tables(retrieval_result)
        if self._has_malformed_tokens(sql_clean, known_tables):
            return False

        return any(
            re.search(rf"\b{re.escape(table)}\b", sql_clean, re.IGNORECASE)
            for table in known_tables
        )

    def _has_malformed_tokens(self, sql: str, known_tables: set[str]) -> bool:
        table_pattern = "|".join(re.escape(table) for table in sorted(known_tables))
        malformed_pattern = re.compile(
            rf"\b({SQL_KEYWORDS})(?=({SQL_KEYWORDS})\b|[A-Za-z_][A-Za-z0-9_]*\.|(?:{table_pattern})\b)",
            re.IGNORECASE,
        )
        return bool(malformed_pattern.search(sql))

    def _known_tables(self, retrieval_result: dict) -> set[str]:
        tables = set(DEFAULT_KNOWN_TABLES)

        for item in retrieval_result.get("schema_docs", []):
            table_name = item.get("table_name")
            if table_name:
                tables.add(table_name)

        return tables

    def _fallback_sql(self, retrieval_result: dict) -> str:
        templates = retrieval_result.get("sql_templates", [])
        if templates:
            return templates[0]["sql"]
        raise ValueError("Generated SQL failed validation and no fallback template is available.")
