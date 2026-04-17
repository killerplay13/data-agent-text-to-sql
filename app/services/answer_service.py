from langchain_openai import ChatOpenAI
from app.core.config import settings


class AnswerService:
    def __init__(self):
        self.llm = None

        if settings.OPENROUTER_API_KEY:
            self.llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.OPENROUTER_MODEL,
                temperature=0
            )

    def build_prompt(self, user_query: str, query_result: list[dict]) -> str:
        return f"""
You are a banking data assistant.

Your task is to answer the user's original question in natural language based only on the SQL query result provided.

Rules:
1. Be concise and clear.
2. Do not invent facts.
3. Only use the provided query result.
4. If the query result is empty, say that no matching data was found.

User Question:
{user_query}

Query Result:
{query_result}

Now provide the final answer in natural language only.
""".strip()

    def fallback_answer(self, user_query: str, query_result: list[dict]) -> str:
        if not query_result:
            return "No matching data was found for the query."

        if len(query_result) == 1:
            return f"Found 1 row. {self._format_row(query_result[0])}."

        preview_limit = 3
        preview_rows = query_result[:preview_limit]
        preview = "\n".join(
            f"{index}. {self._format_row(row)}"
            for index, row in enumerate(preview_rows, start=1)
        )
        remaining_count = len(query_result) - len(preview_rows)
        remaining_text = (
            f"\n...and {remaining_count} more row(s)."
            if remaining_count
            else ""
        )

        return (
            f"Found {len(query_result)} rows. "
            f"Preview of the first {len(preview_rows)} row(s):\n"
            f"{preview}"
            f"{remaining_text}"
        )

    def _format_row(self, row: dict) -> str:
        if not row:
            return "empty row"

        return ", ".join(
            f"{key}: {self._format_value(value)}"
            for key, value in row.items()
        )

    def _format_value(self, value) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        return str(value)

    def generate_answer(self, user_query: str, query_result: list[dict]) -> str:
        if not self.llm:
            return self.fallback_answer(user_query, query_result)

        prompt = self.build_prompt(user_query, query_result)

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"LLM answer generation failed, fallback to rule-based answer. Reason: {e}")
            return self.fallback_answer(user_query, query_result)
