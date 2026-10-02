"""Shared RAG settings — single source of truth."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ONNX via FastEmbed — стабильно на Windows (без PyTorch-crash 0xC0000005)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

MAX_SEMANTIC_DISTANCE = 0.55

RETRIEVE_TOP_K = 12
SEMANTIC_TOP_K = 12
FINAL_TOP_K = 4

GROQ_MODEL = "llama-3.3-70b-versatile"

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

INDEX_BATCH_SIZE = 32
