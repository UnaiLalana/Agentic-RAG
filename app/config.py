import os


class Config:
    """Application configuration from environment variables."""

    # MinIO
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # ChromaDB
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_chunks")

    # LLM
    LLM_URL = os.getenv("LLM_URL", "http://localhost:8080")

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # SQLite metadata DB
    DB_PATH = os.getenv("DB_PATH", "/app/data/metadata.db")
