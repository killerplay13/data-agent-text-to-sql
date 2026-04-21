"""Build true OpenSearch vector indices for the knowledge base.

This script recreates the three KB indices, preserves the original source
fields for backward-compatible keyword search, and adds:
- content: normalized retrieval text
- embedding: knn_vector for semantic search
- doc_type: stable source category
- source_id: stable source identifier
"""

from pathlib import Path
import json

from opensearchpy import OpenSearch

from app.core.config import settings
from app.services.embedding_service import EmbeddingService


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"


def load_json(file_name: str):
    path = KB_DIR / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def build_schema_content(document: dict) -> str:
    column_parts = []
    for column in document.get("columns", []):
        column_parts.append(
            f"{column['name']} ({column['data_type']}): {column['description']}"
        )

    content = "\n".join(
        [
            f"Schema table: {document.get('table_name', '')}",
            f"Description: {document.get('description', '')}",
            f"Columns: {'; '.join(column_parts)}",
        ]
    )
    return normalize_text(content)


def build_sql_template_content(document: dict) -> str:
    content = "\n".join(
        [
            f"SQL template: {document.get('name', '')}",
            f"Example question: {document.get('question_example', '')}",
            f"Business description: {document.get('business_description', '')}",
            f"SQL: {document.get('sql', '')}",
        ]
    )
    return normalize_text(content)


def build_business_context_content(document: dict) -> str:
    content = "\n".join(
        [
            f"Business topic: {document.get('topic', '')}",
            f"Description: {document.get('description', '')}",
        ]
    )
    return normalize_text(content)


INDEX_DEFINITIONS = [
    {
        "file_name": "schema_docs.json",
        "index_name": settings.OPENSEARCH_INDEX_SCHEMA,
        "doc_type": "schema_doc",
        "source_id_field": "table_name",
        "content_builder": build_schema_content,
        "mappings": {
            "type": {"type": "keyword"},
            "table_name": {"type": "keyword"},
            "description": {"type": "text"},
            "columns": {
                "type": "object",
                "properties": {
                    "name": {"type": "text"},
                    "data_type": {"type": "keyword"},
                    "description": {"type": "text"},
                },
            },
            "tags": {"type": "keyword"},
        },
    },
    {
        "file_name": "sql_templates.json",
        "index_name": settings.OPENSEARCH_INDEX_SQL,
        "doc_type": "sql_template",
        "source_id_field": "name",
        "content_builder": build_sql_template_content,
        "mappings": {
            "type": {"type": "keyword"},
            "name": {"type": "keyword"},
            "question_example": {"type": "text"},
            "sql": {"type": "text"},
            "business_description": {"type": "text"},
            "tags": {"type": "keyword"},
        },
    },
    {
        "file_name": "business_context.json",
        "index_name": settings.OPENSEARCH_INDEX_CONTEXT,
        "doc_type": "business_context",
        "source_id_field": "topic",
        "content_builder": build_business_context_content,
        "mappings": {
            "type": {"type": "keyword"},
            "topic": {"type": "keyword"},
            "description": {"type": "text"},
            "tags": {"type": "keyword"},
        },
    },
]


def build_index_body(extra_mappings: dict) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "doc_type": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.EMBEDDING_DIMENSION,
                },
                **extra_mappings,
            },
        },
    }


def recreate_index(client: OpenSearch, index_name: str, extra_mappings: dict):
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)

    print(
        f"Creating vector index: {index_name} "
        f"(dimension={settings.EMBEDDING_DIMENSION})"
    )
    client.indices.create(
        index=index_name,
        body=build_index_body(extra_mappings),
    )


def build_index_documents(
    raw_documents: list[dict],
    doc_type: str,
    source_id_field: str,
    content_builder,
    embedding_service: EmbeddingService,
) -> list[dict]:
    contents = [content_builder(document) for document in raw_documents]
    print(f"Generating embeddings for {len(contents)} {doc_type} documents...")
    embeddings = embedding_service.embed_texts(contents)

    indexed_documents = []
    for position, (document, content, embedding) in enumerate(
        zip(raw_documents, contents, embeddings)
    ):
        source_id = str(document.get(source_id_field) or position)
        indexed_documents.append(
            {
                **document,
                "doc_type": doc_type,
                "source_id": source_id,
                "content": content,
                "embedding": embedding,
            }
        )

    return indexed_documents


def index_documents(client: OpenSearch, index_name: str, documents: list[dict]):
    print(f"Indexing {len(documents)} documents into {index_name}...")

    for document in documents:
        client.index(
            index=index_name,
            id=document["source_id"],
            body=document,
        )

    client.indices.refresh(index=index_name)
    print(f"Finished indexing {len(documents)} documents into {index_name}.")


def main():
    print("Starting vector index build...")
    client = OpenSearch(hosts=[settings.OPENSEARCH_URL])
    embedding_service = EmbeddingService()

    for definition in INDEX_DEFINITIONS:
        raw_documents = load_json(definition["file_name"])
        recreate_index(
            client,
            definition["index_name"],
            definition["mappings"],
        )
        indexed_documents = build_index_documents(
            raw_documents=raw_documents,
            doc_type=definition["doc_type"],
            source_id_field=definition["source_id_field"],
            content_builder=definition["content_builder"],
            embedding_service=embedding_service,
        )
        index_documents(client, definition["index_name"], indexed_documents)

    print("Vector index build complete.")


if __name__ == "__main__":
    main()
