from flask import Blueprint, request, jsonify

from services.retriever import RetrieverService
from services.llm_service import LLMService

query_bp = Blueprint("query", __name__)

# Lazy-initialized singletons
_retriever: RetrieverService = None
_llm: LLMService = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RetrieverService()
    return _retriever


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


@query_bp.route("/query", methods=["POST"])
def query_documents():
    """
    POST /query
    Submit a question. Returns the generated answer and the source chunks used.

    Request body:
        {"question": "What is ...?", "top_k": 5}

    Response:
        {
            "answer": "Based on [1]...",
            "question": "What is ...?",
            "sources": [
                {"text": "...", "doc_id": "...", "filename": "...", "score": 0.85}
            ]
        }
    """
    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({
            "error": "Request body must contain a 'question' field.",
            "example": {"question": "What is the main topic of the document?"}
        }), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    top_k = data.get("top_k", 5)

    # 1. Retrieve relevant chunks
    try:
        chunks = _get_retriever().search(question, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # 2. Generate answer using LLM
    try:
        answer = _get_llm().generate_answer(question, chunks)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    # 3. Format source chunks for response
    sources = [
        {
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "relevance_score": chunk["score"],
        }
        for chunk in chunks
    ]

    return jsonify({
        "answer": answer,
        "question": question,
        "sources": sources,
        "num_sources": len(sources),
    }), 200
