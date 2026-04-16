import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    RETRIEVAL_BACKEND: str = os.getenv("RETRIEVAL_BACKEND", "opensearch")
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_URL: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    OPENSEARCH_INDEX_SCHEMA: str = os.getenv("OPENSEARCH_INDEX_SCHEMA", "schema_docs")
    OPENSEARCH_INDEX_SQL: str = os.getenv("OPENSEARCH_INDEX_SQL", "sql_templates")
    OPENSEARCH_INDEX_CONTEXT: str = os.getenv("OPENSEARCH_INDEX_CONTEXT", "business_context")


settings = Settings()
