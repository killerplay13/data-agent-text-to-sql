import re
import json
from pathlib import Path

from app.services.sql_generation_service import SQLGenerationService
from app.skills.base_skill import BaseSkill


BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "kb"
FULL_SCHEMA_DOCS_PATH = KB_DIR / "schema_docs.json"
TABLE_REFERENCE_PATTERN = re.compile(
    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate)\b",
    re.IGNORECASE,
)
QUOTED_TEXT_PATTERN = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")


class SQLSkill(BaseSkill):
    name = "sql_generation"

    def __init__(self):
        self.service = SQLGenerationService()
        self._full_schema_tables = self._load_full_schema_tables()

    def execute(self, input: dict) -> dict:
        input["generated_sql"] = self.service.generate_sql(
            input["user_query"],
            input["retrieval_result"],
        )

        validation_error = self._validation_error(
            input["generated_sql"],
            input["retrieval_result"],
        )
        if validation_error:
            print("Generated SQL failed validation; attempting repair before fallback.")
            repaired_sql = self.service.repair_sql(
                input["user_query"],
                input["retrieval_result"],
                input["generated_sql"],
                validation_error,
            )
            if repaired_sql and not self._validation_error(
                repaired_sql,
                input["retrieval_result"],
            ):
                input["generated_sql"] = repaired_sql
            else:
                fallback_sql = self._fallback_sql(input["retrieval_result"])
                if fallback_sql:
                    input["generated_sql"] = fallback_sql
                else:
                    raise ValueError(
                        "Generated SQL failed validation, repair did not succeed, "
                        "and no fallback SQL template was retrieved."
                    )

        return input

    def _is_valid_sql(self, sql: str, retrieval_result: dict) -> bool:
        return self._validation_error(sql, retrieval_result) is None

    def _validation_error(self, sql: str, retrieval_result: dict) -> str | None:
        if not sql or not isinstance(sql, str):
            return "Generated SQL is empty or not a string."

        sql_clean = sql.strip()
        sql_lower = sql_clean.lower()

        if not sql_lower.startswith("select"):
            return "Generated SQL must start with SELECT."
        if self._has_dangerous_keywords(sql_clean):
            return "Generated SQL contains disallowed non-SELECT keywords."

        allowed_tables = self._allowed_tables(retrieval_result)
        used_tables = self._extract_tables(sql_clean)

        if not used_tables:
            return "Generated SQL does not reference any tables in FROM or JOIN clauses."

        if allowed_tables and not used_tables.issubset(allowed_tables):
            invalid_tables = sorted(used_tables - allowed_tables)
            return (
                "Generated SQL references tables outside the allowed schema: "
                + ", ".join(invalid_tables)
            )

        return None

    def _has_dangerous_keywords(self, sql: str) -> bool:
        return bool(DANGEROUS_SQL_PATTERN.search(self._strip_quoted_text(sql)))

    def _extract_tables(self, sql: str) -> set[str]:
        stripped_sql = self._strip_quoted_text(sql)
        return {
            match.group(1).lower()
            for match in TABLE_REFERENCE_PATTERN.finditer(stripped_sql)
        }

    def _allowed_tables(self, retrieval_result: dict) -> set[str]:
        schema_docs = retrieval_result.get("schema_docs", [])
        retrieved_tables = {
            item["table_name"].lower()
            for item in schema_docs
            if item.get("table_name")
        }
        return retrieved_tables.union(self._full_schema_tables)

    def _load_full_schema_tables(self) -> set[str]:
        with open(FULL_SCHEMA_DOCS_PATH, "r", encoding="utf-8") as file:
            schema_docs = json.load(file)

        return {
            item["table_name"].lower()
            for item in schema_docs
            if item.get("table_name")
        }

    def _strip_quoted_text(self, sql: str) -> str:
        return QUOTED_TEXT_PATTERN.sub("", sql)

    def _fallback_sql(self, retrieval_result: dict) -> str | None:
        return self.service.fallback_sql(retrieval_result)
