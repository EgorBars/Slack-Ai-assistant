"""Lightweight embeddings module — no PyTorch, safe for app.py."""

from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

from rag_config import EMBEDDING_MODEL

_instance = None


class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


def get_embeddings() -> Embeddings:
    global _instance
    if _instance is None:
        print(f"Loading embeddings model '{EMBEDDING_MODEL}' (ONNX)...")
        _instance = FastEmbedWrapper(EMBEDDING_MODEL)
        print("Embeddings ready.")
    return _instance
