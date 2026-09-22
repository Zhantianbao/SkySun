"""SkySun embedding boundary and local multilingual E5 adapter."""

from typing import Protocol, Sequence

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class SentenceTransformerEmbeddingProvider:
    """Load one CPU model instance and apply E5 retrieval prefixes."""

    dimension = EMBEDDING_DIMENSION

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception:
            raise RuntimeError("Local embedding model could not be loaded") from None
        actual = self._model.get_embedding_dimension()
        if actual != self.dimension:
            raise RuntimeError(
                f"Embedding model dimension {actual} does not match database dimension {self.dimension}"
            )

    def _encode(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        try:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            raise RuntimeError("Local embedding inference failed") from None
        return tuple(tuple(float(value) for value in vector) for vector in vectors)

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("Knowledge search query must not be empty")
        return self._encode([f"query: {text}"])[0]
