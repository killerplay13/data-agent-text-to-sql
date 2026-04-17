import re

from langchain_openai import ChatOpenAI
from app.core.config import settings


SQL_CODE_FENCE_PATTERN = re.compile(
    r"```(?:sql|sqlite)?\s*(?P<sql>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
SQL_SELECT_PATTERN = re.compile(r"\bselect\b", re.IGNORECASE)
SQL_KEYWORD_SPACING_PATTERN = re.compile(
    r"([A-Za-z0-9_)\]])"
    r"(SELECT\b|FROM\b|WHERE\b|GROUP\s+BY\b|ORDER\s+BY\b|HAVING\b|"
    r"LIMIT\b|JOIN\b|INNER\s+JOIN\b|LEFT\s+JOIN\b|RIGHT\s+JOIN\b|"
    r"FULL\s+JOIN\b|CROSS\s+JOIN\b|ON\b)",
    re.IGNORECASE,
)
SQL_ALIAS_SPACING_PATTERN = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|ON|BY|HAVING|LIMIT)(?=[A-Za-z_][A-Za-z0-9_]*\.)",
    re.IGNORECASE,
)
QUOTED_SQL_TEXT_PATTERN = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")


class SQLGenerationService:
    def __init__(self):
        self.llm = None

        if settings.OPENROUTER_API_KEY:
            self.llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.OPENROUTER_MODEL,
                temperature=0
            )

    def build_prompt(self, user_query: str, retrieval_result: dict) -> str:
        schema_text = "\n\n".join(
            [
                f"Table: {item['table_name']}\nDescription: {item['description']}\nColumns: "
                + ", ".join([col["name"] for col in item["columns"]])
                for item in retrieval_result["schema_docs"]
            ]
        )

        template_text = "\n\n".join(
            [
                f"Template Name: {item['name']}\nExample Question: {item['question_example']}\nSQL: {item['sql']}\nBusiness Description: {item['business_description']}"
                for item in retrieval_result["sql_templates"]
            ]
        )

        context_text = "\n\n".join(
            [
                f"Topic: {item['topic']}\nDescription: {item['description']}"
                for item in retrieval_result["business_context"]
            ]
        )

        return f"""
You are a senior Text-to-SQL assistant.

Your task is to generate a single valid SQLite SELECT query based on the user's question.

Rules:
1. Only generate SQL.
2. Only generate ONE query.
3. Only use the tables and columns provided below.
4. Do not invent tables or columns.
5. Only output a SELECT query. Never output DELETE, UPDATE, INSERT, DROP, ALTER, or explanations.
6. The target database is SQLite.
7. Output exactly one SQL SELECT query and nothing else.
8. Do not include explanations.
9. Do not include markdown.
10. Do not include code fences.
11. Do not include bullet points.
12. The response must start with SELECT.
13. The retrieved SQL templates are the PRIMARY reference.
14. You MUST follow the structure of the closest SQL template.
15. If a SQL template is relevant, you MUST reuse its joins and structure.
16. Do not generate a new SQL query from scratch.
17. Do not simplify the query by removing joins or filters.
18. Do not change the business logic, ranking, filtering, or aggregation intent of the template.
19. Only adapt filters, limits, or ordering when explicitly required by the user's question.
20. Do not introduce SUM, AVG, COUNT, or GROUP BY unless clearly needed by the user's question.

User Question:
{user_query}

Relevant Schema:
{schema_text}

Relevant SQL Templates:
{template_text}

Relevant Business Context:
{context_text}

Now generate the SQL query only.
""".strip()

    def fallback_sql(self, retrieval_result: dict) -> str:
        templates = retrieval_result.get("sql_templates", [])
        if templates:
            return templates[0]["sql"]
        raise ValueError("No SQL template available for fallback.")

    def clean_generated_sql(self, content: str) -> str:
        sql = self._extract_sql(content)

        if not sql:
            return ""

        return self._restore_keyword_spacing(sql)

    def _extract_sql(self, content: str) -> str:
        raw_content = content.strip()
        fenced_sql = SQL_CODE_FENCE_PATTERN.search(raw_content)

        if fenced_sql:
            return fenced_sql.group("sql").strip()

        select_match = SQL_SELECT_PATTERN.search(raw_content)
        if not select_match:
            return ""

        sql = raw_content[select_match.start():]
        final_semicolon_index = sql.rfind(";")
        if final_semicolon_index != -1:
            sql = sql[:final_semicolon_index + 1]

        return sql.strip()

    def _restore_keyword_spacing(self, sql: str) -> str:
        parts = QUOTED_SQL_TEXT_PATTERN.split(sql)

        for index in range(0, len(parts), 2):
            parts[index] = SQL_KEYWORD_SPACING_PATTERN.sub(r"\1 \2", parts[index])
            parts[index] = SQL_ALIAS_SPACING_PATTERN.sub(r"\1 ", parts[index])

        return "".join(parts)

    def generate_sql(self, user_query: str, retrieval_result: dict) -> str:
        if not self.llm:
            return self.fallback_sql(retrieval_result)

        prompt = self.build_prompt(user_query, retrieval_result)

        try:
            response = self.llm.invoke(prompt)
            print(f"Raw LLM SQL response: {response.content}")
            sql = self.clean_generated_sql(response.content)
            print(f"Cleaned SQL: {sql}")
            if not sql or not sql.lower().startswith("select"):
                return self.fallback_sql(retrieval_result)
            return sql
        except Exception as e:
            print(f"LLM generation failed, fallback to template. Reason: {e}")
            return self.fallback_sql(retrieval_result)
