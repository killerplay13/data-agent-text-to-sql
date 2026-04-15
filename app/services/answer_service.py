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
            return "No matching data was found."

        first_row = query_result[0]

        if "customer_name" in first_row and "deposit_amount" in first_row:
            amount = f"{first_row['deposit_amount']:,.0f}"
            return f"The customer with the highest deposit is {first_row['customer_name']}, with a deposit amount of {amount}."

        return f"Query completed successfully. Result: {query_result}"

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