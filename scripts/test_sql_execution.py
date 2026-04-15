from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService
from app.services.execution_service import ExecutionService


def main():
    user_query = "Who has the highest deposit?"

    retrieval_service = RetrievalService()
    sql_service = SQLGenerationService()
    execution_service = ExecutionService()

    retrieval_result = retrieval_service.retrieve(user_query)
    generated_sql = sql_service.generate_sql(user_query, retrieval_result)
    query_result = execution_service.execute_query(generated_sql)

    print("User Query:")
    print(user_query)

    print("\nGenerated SQL:")
    print(generated_sql)

    print("\nQuery Result:")
    for row in query_result:
        print(row)


if __name__ == "__main__":
    main()