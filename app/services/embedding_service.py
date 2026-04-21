from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingServiceError(RuntimeError):
    """Raised when embedding generation fails or returns invalid vectors."""


class EmbeddingService:
    def __init__(self):
        self._model = None

    def embed_text(self, text: str) -> list[float]:
        """Generate a single embedding and validate its configured dimension."""
        if not text.strip():
            raise EmbeddingServiceError("Cannot generate an embedding for empty text.")

        try:
            embedding = self._get_model().encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Embedding generation failed for a single text input: {exc}"
            ) from exc

        return self._validate_embedding(embedding)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts and validate each vector length."""
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise EmbeddingServiceError(
                "Cannot generate embeddings when one or more text inputs are empty."
            )

        try:
            embeddings = self._get_model().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Embedding generation failed for batched text inputs: {exc}"
            ) from exc

        return [self._validate_embedding(embedding) for embedding in embeddings]

    def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        if settings.EMBEDDING_PROVIDER not in {"sentence_transformers", "local"}:
            raise EmbeddingServiceError(
                "Unsupported embedding provider "
                f"'{settings.EMBEDDING_PROVIDER}'. Expected 'sentence_transformers'."
            )

        try:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Failed to load local embedding model '{settings.EMBEDDING_MODEL}': {exc}"
            ) from exc

        model_dimension = self._model.get_embedding_dimension()
        if model_dimension != settings.EMBEDDING_DIMENSION:
            raise EmbeddingServiceError(
                "Embedding dimension mismatch between config and local model: "
                f"expected {settings.EMBEDDING_DIMENSION}, got {model_dimension} "
                f"for '{settings.EMBEDDING_MODEL}'."
            )

        return self._model

    def _validate_embedding(self, embedding: list[float]) -> list[float]:
        if len(embedding) != settings.EMBEDDING_DIMENSION:
            raise EmbeddingServiceError(
                "Embedding dimension mismatch: "
                f"expected {settings.EMBEDDING_DIMENSION}, got {len(embedding)}."
            )

        return embedding
