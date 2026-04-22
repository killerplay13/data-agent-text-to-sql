import re

from app.services.execution_service import ExecutionService
from app.services.sql_generation_service import SQLGenerationService
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
        self.sql_service = SQLGenerationService()

    def execute(self, input: dict) -> dict:
        retrieval_result = input.get("retrieval_result", {})
        validation_error = self._validation_error(input["generated_sql"], retrieval_result)
        if validation_error:
            repaired_sql = self.sql_service.repair_sql(
                input["user_query"],
                retrieval_result,
                input["generated_sql"],
                validation_error,
            )
            if repaired_sql and self._is_valid_sql(repaired_sql, retrieval_result):
                input["generated_sql"] = repaired_sql
            else:
                fallback_sql = self._fallback_sql(retrieval_result)
                if fallback_sql:
                    input["generated_sql"] = fallback_sql
                else:
                    raise ValueError(
                        "Generated SQL failed execution-layer validation, repair did not "
                        "succeed, and no fallback SQL template was retrieved."
                    )

        try:
            input["query_result"] = self.service.execute_query(input["generated_sql"])
        except Exception as e:
            repaired_sql = self.sql_service.repair_sql(
                input["user_query"],
                retrieval_result,
                input["generated_sql"],
                f"SQL execution error: {e}",
            )
            if repaired_sql and self._is_valid_sql(repaired_sql, retrieval_result):
                input["generated_sql"] = repaired_sql
                input["query_result"] = self.service.execute_query(input["generated_sql"])
            else:
                fallback_sql = self._fallback_sql(retrieval_result)
                if not fallback_sql:
                    raise ValueError(
                        "SQL execution failed, repair did not succeed, "
                        "and no fallback SQL template was retrieved."
                    ) from e
                input["generated_sql"] = fallback_sql
                input["query_result"] = self.service.execute_query(input["generated_sql"])
        return input

    def _is_valid_sql(self, sql: str, retrieval_result: dict) -> bool:
        return self._validation_error(sql, retrieval_result) is None

    def _validation_error(self, sql: str, retrieval_result: dict) -> str | None:
        if not isinstance(sql, str):
            return "Generated SQL is not a string."

        sql_clean = sql.strip()
        if not sql_clean.lower().startswith("select"):
            return "Generated SQL must start with SELECT."

        known_tables = self._known_tables(retrieval_result)
        if self._has_malformed_tokens(sql_clean, known_tables):
            return "Generated SQL contains malformed SQL keyword spacing."

        has_known_table = any(
            re.search(rf"\b{re.escape(table)}\b", sql_clean, re.IGNORECASE)
            for table in known_tables
        )
        if not has_known_table:
            return "Generated SQL does not reference a known schema table."

        return None

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
        return self.sql_service.fallback_sql(retrieval_result)
