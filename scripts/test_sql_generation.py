from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService


def main():
    user_query = "Who has the highest deposit?"

    retrieval_service = RetrievalService()
    retrieval_result = retrieval_service.retrieve(user_query)

    sql_service = SQLGenerationService()
    generated_sql = sql_service.generate_sql(user_query, retrieval_result)

    print("User Query:")
    print(user_query)

    print("\nGenerated SQL:")
    print(generated_sql)


if __name__ == "__main__":
    main()