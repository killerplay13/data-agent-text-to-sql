from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService
from app.services.execution_service import ExecutionService
from app.services.answer_service import AnswerService


def main():
    user_query = "Who has the highest deposit?"

    retrieval_service = RetrievalService()
    sql_service = SQLGenerationService()
    execution_service = ExecutionService()
    answer_service = AnswerService()

    retrieval_result = retrieval_service.retrieve(user_query)
    generated_sql = sql_service.generate_sql(user_query, retrieval_result)
    query_result = execution_service.execute_query(generated_sql)
    answer = answer_service.generate_answer(user_query, query_result)

    print("User Query:")
    print(user_query)

    print("\nGenerated SQL:")
    print(generated_sql)

    print("\nQuery Result:")
    print(query_result)

    print("\nFinal Answer:")
    print(answer)


if __name__ == "__main__":
    main()