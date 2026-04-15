from app.services.retrieval_service import RetrievalService


def main():
    service = RetrievalService()

    query = "Who has the highest deposit?"

    result = service.retrieve(query)

    print("\n=== SQL Templates ===")
    for item in result["sql_templates"]:
        print("-", item["name"])

    print("\n=== Schema Docs ===")
    for item in result["schema_docs"]:
        print("-", item["table_name"])

    print("\n=== Business Context ===")
    for item in result["business_context"]:
        print("-", item["topic"])


if __name__ == "__main__":
    main()