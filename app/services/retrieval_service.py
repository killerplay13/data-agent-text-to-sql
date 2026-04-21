from pathlib import Path
import json

from opensearchpy import OpenSearch

from app.core.config import settings
from app.services.embedding_service import EmbeddingService


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
        self.embedding_service = None

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
        if settings.ENABLE_VECTOR_RETRIEVAL:
            if settings.ENABLE_HYBRID_RETRIEVAL:
                print("Using OpenSearch hybrid retrieval.")
                return self._retrieve_from_opensearch_hybrid(
                    user_query,
                    settings.VECTOR_TOP_K,
                )

            print("Using OpenSearch vector retrieval.")
            return self._retrieve_from_opensearch_vector(user_query, settings.VECTOR_TOP_K)

        print("Using OpenSearch keyword retrieval.")
        return self._retrieve_from_opensearch_keyword(user_query, top_k)

    def _retrieve_from_opensearch_keyword(self, user_query: str, top_k: int = 3):
        local_result = self._retrieve_from_local(user_query, top_k)

        try:
            keyword_result = self._retrieve_opensearch_keyword_sources(user_query, top_k)
            return self._apply_local_fallbacks(keyword_result, local_result, "keyword")
        except Exception as e:
            print(f"OpenSearch retrieval failed, falling back to local retrieval. Reason: {e}")
            return local_result

    def _retrieve_from_opensearch_vector(self, user_query: str, top_k: int):
        local_result = self._retrieve_from_local(user_query, top_k)

        try:
            vector_result = self._retrieve_opensearch_vector_sources(user_query, top_k)
            return self._apply_local_fallbacks(vector_result, local_result, "vector")
        except Exception as e:
            print(f"OpenSearch vector retrieval failed, falling back to local retrieval. Reason: {e}")
            return local_result

    def _retrieve_from_opensearch_hybrid(self, user_query: str, top_k: int):
        local_result = self._retrieve_from_local(user_query, top_k)

        try:
            vector_result = self._retrieve_opensearch_vector_sources(user_query, top_k)
            keyword_result = self._retrieve_opensearch_keyword_sources(user_query, top_k)

            hybrid_result = {}
            for source_name in self._source_configs():
                hybrid_result[source_name] = self._merge_ranked_results(
                    vector_result[source_name],
                    keyword_result[source_name],
                    top_k,
                )

            return self._apply_local_fallbacks(hybrid_result, local_result, "hybrid")
        except Exception as e:
            print(
                "OpenSearch hybrid retrieval failed, falling back to vector retrieval. "
                f"Reason: {e}"
            )
            return self._retrieve_from_opensearch_vector(user_query, top_k)

    def _retrieve_opensearch_keyword_sources(
        self,
        user_query: str,
        top_k: int,
    ) -> dict[str, list[dict]]:
        keyword_result = {}

        for source_name, config in self._source_configs().items():
            query = self._build_keyword_query(user_query, config["keyword_fields"], top_k)
            response = self._get_opensearch_client().search(
                index=config["index_name"],
                body=query,
            )
            print(f"{source_name} keyword hits: {len(response['hits']['hits'])}")
            keyword_result[source_name] = self._extract_hit_sources(response)

        return keyword_result

    def _retrieve_opensearch_vector_sources(
        self,
        user_query: str,
        top_k: int,
    ) -> dict[str, list[dict]]:
        query_embedding = self._get_embedding_service().embed_text(user_query)
        vector_result = {}

        for source_name, config in self._source_configs().items():
            response = self._search_knn_index(
                config["index_name"],
                query_embedding,
                top_k,
            )
            print(f"{source_name} vector hits: {len(response['hits']['hits'])}")
            vector_result[source_name] = self._extract_hit_sources(response)

        return vector_result

    def _search_knn_index(self, index_name: str, query_embedding: list[float], top_k: int):
        knn_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
        }
        return self._get_opensearch_client().search(index=index_name, body=knn_query)

    def _build_keyword_query(
        self,
        user_query: str,
        fields: list[str],
        top_k: int,
    ) -> dict:
        return {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": user_query,
                    "fields": fields,
                }
            },
        }

    def _apply_local_fallbacks(
        self,
        retrieved_result: dict[str, list[dict]],
        local_result: dict[str, list[dict]],
        mode_name: str,
    ) -> dict[str, list[dict]]:
        if all(not items for items in retrieved_result.values()):
            print(
                f"{mode_name.capitalize()} retrieval returned no hits. "
                "Falling back to local retrieval."
            )
            return local_result

        final_result = {}
        for source_name, items in retrieved_result.items():
            if not items:
                print(
                    f"No {mode_name} hits for {source_name}. "
                    "Falling back to local retrieval for that source."
                )
                final_result[source_name] = local_result[source_name]
                continue

            final_result[source_name] = items

        return final_result

    def _merge_ranked_results(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        ranked_documents = {}

        self._add_ranked_results(ranked_documents, vector_results, "vector")
        self._add_ranked_results(ranked_documents, keyword_results, "keyword")

        sorted_documents = sorted(
            ranked_documents.values(),
            key=lambda item: (-item["score"], item["first_seen_rank"]),
        )
        return [item["document"] for item in sorted_documents[:top_k]]

    def _add_ranked_results(
        self,
        ranked_documents: dict,
        documents: list[dict],
        source_name: str,
    ):
        for rank, document in enumerate(documents, start=1):
            identity = self._document_identity(document)
            weighted_score = 1 / rank

            if identity not in ranked_documents:
                ranked_documents[identity] = {
                    "document": document,
                    "score": weighted_score,
                    "first_seen_rank": rank,
                    "sources": {source_name},
                }
                continue

            ranked_documents[identity]["score"] += weighted_score
            ranked_documents[identity]["sources"].add(source_name)

    def _document_identity(self, document: dict) -> tuple[str, str]:
        if document.get("name"):
            return ("sql_template", str(document["name"]))

        if document.get("table_name"):
            return ("schema_doc", str(document["table_name"]))

        if document.get("topic"):
            return ("business_context", str(document["topic"]))

        if document.get("source_id"):
            return ("source_id", str(document["source_id"]))

        return ("document", json.dumps(document, sort_keys=True))

    def _extract_hit_sources(self, response: dict) -> list[dict]:
        sources = []

        for hit in response["hits"]["hits"]:
            source = dict(hit["_source"])
            source.pop("embedding", None)
            source.pop("content", None)
            source.pop("doc_type", None)
            source.pop("source_id", None)
            sources.append(source)

        return sources

    def _get_opensearch_client(self) -> OpenSearch:
        if self.opensearch_client is None:
            self.opensearch_client = OpenSearch(hosts=[settings.OPENSEARCH_URL])

        return self.opensearch_client

    def _get_embedding_service(self) -> EmbeddingService:
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()

        return self.embedding_service

    def _source_configs(self) -> dict[str, dict]:
        return {
            "sql_templates": {
                "index_name": settings.OPENSEARCH_INDEX_SQL,
                "keyword_fields": ["question_example", "business_description"],
            },
            "schema_docs": {
                "index_name": settings.OPENSEARCH_INDEX_SCHEMA,
                "keyword_fields": ["description", "columns.name"],
            },
            "business_context": {
                "index_name": settings.OPENSEARCH_INDEX_CONTEXT,
                "keyword_fields": ["description"],
            },
        }
