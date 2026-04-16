from pathlib import Path
import json

from opensearchpy import OpenSearch

from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"


def load_json(file_name: str):
    path = KB_DIR / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_index_if_missing(client: OpenSearch, index_name: str):
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)


def index_documents(client: OpenSearch, index_name: str, documents: list[dict], id_field: str):
    create_index_if_missing(client, index_name)

    for position, document in enumerate(documents):
        document_id = document.get(id_field, position)
        client.index(
            index=index_name,
            id=document_id,
            body=document,
            refresh=True,
        )

    print(f"{index_name}: indexed {len(documents)} documents")


def main():
    client = OpenSearch(hosts=[settings.OPENSEARCH_URL])

    schema_docs = load_json("schema_docs.json")
    sql_templates = load_json("sql_templates.json")
    business_context = load_json("business_context.json")

    index_documents(
        client,
        settings.OPENSEARCH_INDEX_SCHEMA,
        schema_docs,
        "table_name",
    )
    index_documents(
        client,
        settings.OPENSEARCH_INDEX_SQL,
        sql_templates,
        "name",
    )
    index_documents(
        client,
        settings.OPENSEARCH_INDEX_CONTEXT,
        business_context,
        "topic",
    )


if __name__ == "__main__":
    main()
