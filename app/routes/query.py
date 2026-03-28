from flask import Blueprint, request, jsonify

from services.retriever import RetrieverService
from services.llm_service import LLMService
from services.source_searcher import SourceSearcher

query_bp = Blueprint("query", __name__)

# Lazy-initialized singletons
_retriever: RetrieverService = None
_llm: LLMService = None
_searcher: SourceSearcher = None


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


def _get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = SourceSearcher()
    return _searcher


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
                {"text": "...", "doc_id": "...", "filename": "...", "score": 0.85, "web_source": "https://..."}
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
        # Simple intent routing: if the user asks for plagiarized or internet-copied content,
        # explicitly fetch chunks that have an internet source_url.
        question_lower = question.lower()
        plagiarism_keywords = ["copi", "pegad", "plagio", "internet", "copy", "paste", "plagiar", "fuente", "web"]
        is_plagiarism_query = any(k in question_lower for k in plagiarism_keywords)

        if is_plagiarism_query:
            chunks = _get_retriever().get_chunks_with_sources(top_k=top_k)
            # If nothing found, fallback to semantic search
            if not chunks:
                chunks = _get_retriever().search(question, top_k=top_k)
        else:
            chunks = _get_retriever().search(question, top_k=top_k)

    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500
        
    # 1.5. Propagate web sources from ChromaDB (already found during upload)
    for chunk in chunks:
        chunk["web_source"] = chunk.get("source_url", "")

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
            "web_source": chunk.get("web_source", "")
        }
        for chunk in chunks
    ]

    return jsonify({
        "answer": answer,
        "question": question,
        "sources": sources,
        "num_sources": len(sources),
    }), 200
