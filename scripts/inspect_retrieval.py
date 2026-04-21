import argparse

from app.core.config import settings
from app.services.retrieval_service import RetrievalService


def get_active_retrieval_mode() -> str:
    if settings.RETRIEVAL_BACKEND != "opensearch":
        return "local"

    if settings.ENABLE_VECTOR_RETRIEVAL and settings.ENABLE_HYBRID_RETRIEVAL:
        return "opensearch_hybrid"

    if settings.ENABLE_VECTOR_RETRIEVAL:
        return "opensearch_vector"

    return "opensearch_keyword"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval results for a natural language query.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query to inspect.",
    )
    return parser.parse_args()


def prompt_for_query() -> str:
    return input("Enter a natural language query: ").strip()


def print_section(title: str, items: list[dict]):
    print(f"\n=== {title} ===")

    if not items:
        print("(no results)")
        return

    for index, item in enumerate(items, start=1):
        identifier = (
            item.get("name")
            or item.get("table_name")
            or item.get("topic")
            or "unknown"
        )
        summary = (
            item.get("business_description")
            or item.get("description")
            or item.get("question_example")
            or "(no summary)"
        )
        tags = item.get("tags")

        print(f"{index}. {identifier}")
        print(f"   Summary: {summary}")
        if tags:
            print(f"   Tags: {', '.join(tags)}")


def main():
    args = parse_args()
    query = (args.query or "").strip()

    if not query:
        query = prompt_for_query()

    if not query:
        raise ValueError("A query is required to inspect retrieval results.")

    print(f"Retrieval backend: {settings.RETRIEVAL_BACKEND}")
    print(f"Active retrieval mode: {get_active_retrieval_mode()}")
    print(f"Query: {query}")

    service = RetrievalService()
    result = service.retrieve(query)

    print_section("SQL Templates", result.get("sql_templates", []))
    print_section("Schema Docs", result.get("schema_docs", []))
    print_section("Business Context", result.get("business_context", []))


if __name__ == "__main__":
    main()
