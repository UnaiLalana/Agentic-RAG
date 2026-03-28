import uuid
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from database import get_db
from services.minio_service import MinioService
from services.parser import parse_document, ParserError
from services.chunker import recursive_split
from services.retriever import RetrieverService
from services.ai_detector import AIDetector
from services.source_searcher import SourceSearcher

documents_bp = Blueprint("documents", __name__)

# Lazy-initialized singletons
_minio: MinioService = None
_retriever: RetrieverService = None


def _get_minio():
    global _minio
    if _minio is None:
        _minio = MinioService()
    return _minio


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RetrieverService()
    return _retriever


ALLOWED_EXTENSIONS = {"pdf", "docx"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@documents_bp.route("/documents", methods=["POST"])
def upload_document():
    """
    POST /documents
    Upload a PDF or DOCX file.
    Returns a document ID and metadata.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' field."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No filename provided."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Read file data
    file_data = file.read()
    filename = file.filename
    doc_id = str(uuid.uuid4())

    # 1. Store in MinIO
    try:
        _get_minio().upload_file(doc_id, filename, file_data)
    except Exception as e:
        return jsonify({"error": f"Failed to store file: {str(e)}"}), 500

    # 2. Parse document
    try:
        text = parse_document(filename, file_data)
    except ParserError as e:
        # Clean up MinIO on parse failure
        _get_minio().delete_document(doc_id)
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        _get_minio().delete_document(doc_id)
        return jsonify({"error": f"Failed to parse document: {str(e)}"}), 500

    # 3. Chunk the text
    chunks = recursive_split(text)

    if not chunks:
        _get_minio().delete_document(doc_id)
        return jsonify({"error": "Document produced no text chunks after processing."}), 422

    # Gather metadata for chunks
    detector = AIDetector()
    searcher = SourceSearcher()
    
    ai_probs = []
    source_urls = []
    
    for chunk in chunks:
        prob = detector.predict_probability(chunk)
        ai_probs.append(prob)
        # Only search if probability is low (e.g. <= 0.2) to find sources for authentic human text
        if prob <= 0.2:
            source = searcher.find_source(chunk)
            source_urls.append(source)
        else:
            source_urls.append("")

    # 4. Index chunks in ChromaDB
    try:
        chunk_count = _get_retriever().index_chunks(
            doc_id, filename, chunks,
            ai_probabilities=ai_probs,
            source_urls=source_urls
        )
    except Exception as e:
        _get_minio().delete_document(doc_id)
        return jsonify({"error": f"Failed to index chunks: {str(e)}"}), 500

    # 5. Store metadata in SQLite
    upload_date = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO documents (id, filename, upload_date, chunk_count, file_size) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, upload_date, chunk_count, len(file_data)),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "id": doc_id,
        "filename": filename,
        "upload_date": upload_date,
        "chunk_count": chunk_count,
        "file_size": len(file_data),
        "message": f"Document uploaded and indexed successfully ({chunk_count} chunks).",
    }), 201


@documents_bp.route("/documents", methods=["GET"])
def list_documents():
    """
    GET /documents
    List all uploaded documents with ID, filename, and upload date.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, upload_date, chunk_count, file_size FROM documents ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()

    documents = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "upload_date": row["upload_date"],
            "chunk_count": row["chunk_count"],
            "file_size": row["file_size"],
        }
        for row in rows
    ]

    return jsonify({"documents": documents, "total": len(documents)}), 200


@documents_bp.route("/documents/<doc_id>/chunks", methods=["GET"])
def get_document_chunks(doc_id):
    """
    GET /documents/{doc_id}/chunks
    Return all indexed chunks for a specific document.
    """
    try:
        # Get from ChromaDB directly or via retriever
        results = _get_retriever().collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        
        if not results or not results["documents"]:
            return jsonify({"chunks": []}), 200

        chunks_data = []
        # Sort by chunk_index
        items = list(zip(results["documents"], results["metadatas"]))
        items.sort(key=lambda x: x[1].get("chunk_index", 0))

        for text, meta in items:
            chunks_data.append({
                "chunk_index": meta.get("chunk_index"),
                "text": text,
                "ai_probability": meta.get("ai_probability", 0.0),
                "source_url": meta.get("source_url", "")
            })
            
        return jsonify({"chunks": chunks_data}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get chunks: {e}"}), 500


@documents_bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """
    DELETE /documents/{id}
    Delete a document and its indexed chunks from all stores.
    """
    # Check if document exists
    conn = get_db()
    row = conn.execute("SELECT id, filename FROM documents WHERE id = ?", (doc_id,)).fetchone()

    if not row:
        conn.close()
        return jsonify({"error": f"Document {doc_id} not found."}), 404

    filename = row["filename"]

    # 1. Remove from ChromaDB
    try:
        _get_retriever().delete_document(doc_id)
    except Exception as e:
        print(f"[Documents] Warning: failed to delete from ChromaDB: {e}")

    # 2. Remove from MinIO
    try:
        _get_minio().delete_document(doc_id)
    except Exception as e:
        print(f"[Documents] Warning: failed to delete from MinIO: {e}")

    # 3. Remove metadata
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

    return jsonify({
        "message": f"Document '{filename}' ({doc_id}) deleted successfully.",
    }), 200
