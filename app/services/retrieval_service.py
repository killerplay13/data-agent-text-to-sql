from pathlib import Path
from app.core.config import settings
import json

from opensearchpy import OpenSearch


BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "kb"


def load_json(file_name: str):
    path = KB_DIR / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RetrievalService:
    def __init__(self):
        self.schema_docs = load_json("schema_docs.json")
        self.sql_templates = load_json("sql_templates.json")
        self.business_context = load_json("business_context.json")
        self.opensearch_client = None

    def _simple_match(self, text: str, query: str) -> int:
        score = 0
        query_words = query.lower().split()

        for word in query_words:
            if word in text.lower():
                score += 1

        return score

    def retrieve(self, user_query: str, top_k: int = 3):
        if settings.RETRIEVAL_BACKEND == "opensearch":
            return self._retrieve_from_opensearch(user_query, top_k)
        else:
            return self._retrieve_from_local(user_query, top_k)

    def _retrieve_from_local(self, user_query: str, top_k: int = 3):
        scored_templates = []
        for item in self.sql_templates:
            text = item["question_example"] + " " + item["business_description"]
            score = self._simple_match(text, user_query)
            scored_templates.append((score, item))

        scored_templates.sort(key=lambda x: x[0], reverse=True)

        scored_schema = []
        for item in self.schema_docs:
            text = item["description"] + " " + " ".join([col["name"] for col in item["columns"]])
            score = self._simple_match(text, user_query)
            scored_schema.append((score, item))

        scored_schema.sort(key=lambda x: x[0], reverse=True)

        scored_context = []
        for item in self.business_context:
            text = item["description"]
            score = self._simple_match(text, user_query)
            scored_context.append((score, item))

        scored_context.sort(key=lambda x: x[0], reverse=True)

        return {
            "sql_templates": [item for _, item in scored_templates[:top_k]],
            "schema_docs": [item for _, item in scored_schema[:top_k]],
            "business_context": [item for _, item in scored_context[:top_k]],
        }

    def _retrieve_from_opensearch(self, user_query: str, top_k: int = 3):
        local_result = self._retrieve_from_local(user_query, top_k)

        try:
            if self.opensearch_client is None:
                self.opensearch_client = OpenSearch(hosts=[settings.OPENSEARCH_URL])

            sql_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": user_query,
                        "fields": ["question_example", "business_description"],
                    }
                },
            }
            schema_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": user_query,
                        "fields": ["description", "columns.name"],
                    }
                },
            }
            context_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": user_query,
                        "fields": ["description"],
                    }
                },
            }

            sql_response = self.opensearch_client.search(
                index=settings.OPENSEARCH_INDEX_SQL,
                body=sql_query,
            )
            print(f"sql_templates hits: {len(sql_response['hits']['hits'])}")
            schema_response = self.opensearch_client.search(
                index=settings.OPENSEARCH_INDEX_SCHEMA,
                body=schema_query,
            )
            print(f"schema_docs hits: {len(schema_response['hits']['hits'])}")
            context_response = self.opensearch_client.search(
                index=settings.OPENSEARCH_INDEX_CONTEXT,
                body=context_query,
            )
            print(f"business_context hits: {len(context_response['hits']['hits'])}")

            sql_templates = [
                hit["_source"] for hit in sql_response["hits"]["hits"]
            ]
            schema_docs = [
                hit["_source"] for hit in schema_response["hits"]["hits"]
            ]
            business_context = [
                hit["_source"] for hit in context_response["hits"]["hits"]
            ]

            if not sql_templates:
                print("fallback to local for sql_templates")
                sql_templates = local_result["sql_templates"]

            if not schema_docs:
                print("fallback to local for schema_docs")
                schema_docs = local_result["schema_docs"]

            if not business_context:
                print("fallback to local for business_context")
                business_context = local_result["business_context"]

            return {
                "sql_templates": sql_templates,
                "schema_docs": schema_docs,
                "business_context": business_context,
            }
        except Exception as e:
            print(f"OpenSearch retrieval failed, falling back to local retrieval. Reason: {e}")
            return local_result
