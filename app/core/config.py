import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    RETRIEVAL_BACKEND: str = os.getenv("RETRIEVAL_BACKEND", "opensearch")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "3"))
    ENABLE_VECTOR_RETRIEVAL: bool = _get_bool_env("ENABLE_VECTOR_RETRIEVAL", "false")
    ENABLE_HYBRID_RETRIEVAL: bool = _get_bool_env("ENABLE_HYBRID_RETRIEVAL", "false")
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_URL: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    OPENSEARCH_INDEX_SCHEMA: str = os.getenv("OPENSEARCH_INDEX_SCHEMA", "schema_docs")
    OPENSEARCH_INDEX_SQL: str = os.getenv("OPENSEARCH_INDEX_SQL", "sql_templates")
    OPENSEARCH_INDEX_CONTEXT: str = os.getenv("OPENSEARCH_INDEX_CONTEXT", "business_context")


settings = Settings()
