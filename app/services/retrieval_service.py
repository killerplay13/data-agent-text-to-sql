from pathlib import Path
import json


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

    def _simple_match(self, text: str, query: str) -> int:
        """
        Very simple keyword matching score
        """
        score = 0
        query_words = query.lower().split()

        for word in query_words:
            if word in text.lower():
                score += 1

        return score

    def retrieve(self, user_query: str, top_k: int = 3):
        """
        Return top_k relevant schema, sql templates, and business context
        """

        # score SQL templates
        scored_templates = []
        for item in self.sql_templates:
            text = item["question_example"] + " " + item["business_description"]
            score = self._simple_match(text, user_query)
            scored_templates.append((score, item))

        scored_templates.sort(key=lambda x: x[0], reverse=True)

        # score schema
        scored_schema = []
        for item in self.schema_docs:
            text = item["description"] + " " + " ".join([col["name"] for col in item["columns"]])
            score = self._simple_match(text, user_query)
            scored_schema.append((score, item))

        scored_schema.sort(key=lambda x: x[0], reverse=True)

        # score business context
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