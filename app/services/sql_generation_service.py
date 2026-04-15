from langchain_openai import ChatOpenAI
from app.core.config import settings


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

    def generate_sql(self, user_query: str, retrieval_result: dict) -> str:
        if not self.llm:
            return self.fallback_sql(retrieval_result)

        prompt = self.build_prompt(user_query, retrieval_result)

        try:
            response = self.llm.invoke(prompt)
            sql = response.content.strip()

            if sql.startswith("```"):
                sql = sql.replace("```sql", "").replace("```", "").strip()

            return sql
        except Exception as e:
            print(f"LLM generation failed, fallback to template. Reason: {e}")
            return self.fallback_sql(retrieval_result)