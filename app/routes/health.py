from flask import Blueprint, jsonify
from services.minio_service import MinioService
from services.retriever import RetrieverService
from services.llm_service import LLMService

health_bp = Blueprint("health", __name__)

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RetrieverService()
    return _retriever


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /health
    Health check confirming all services (MinIO, ChromaDB, LLM) are reachable.
    """
    status = {}

    # Check MinIO
    try:
        minio_svc = MinioService()
        status["minio"] = "ok" if minio_svc.is_healthy() else "error"
    except Exception as e:
        status["minio"] = f"error: {str(e)}"

    # Check ChromaDB
    try:
        retriever = get_retriever()
        status["chromadb"] = "ok" if retriever.is_healthy() else "error"
    except Exception as e:
        status["chromadb"] = f"error: {str(e)}"

    # Check LLM
    try:
        llm = LLMService()
        status["llm"] = "ok" if llm.is_healthy() else "error"
    except Exception as e:
        status["llm"] = f"error: {str(e)}"

    # Overall status
    all_ok = all(v == "ok" for v in status.values())
    status["status"] = "healthy" if all_ok else "degraded"

    return jsonify(status), 200 if all_ok else 503
