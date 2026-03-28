from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import Config


class RetrieverService:
    """
    Bi-Encoder retrieval using sentence-transformers + ChromaDB.

    Uses all-MiniLM-L6-v2 to embed chunks and queries, storing
    vectors in ChromaDB for efficient ANN search.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RetrieverService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: List[str],
        ai_probabilities: Optional[List[float]] = None,
        source_urls: Optional[List[str]] = None,
    ) -> int:
        """
        Embed and index document chunks in ChromaDB.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            chunks: List of text chunks.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()

        # Prepare IDs and metadata
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "ai_probability": float(ai_probabilities[i]) if ai_probabilities else 0.0,
                "source_url": source_urls[i] if source_urls else "",
            }
            for i in range(len(chunks))
        ]

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks for a query.

        Args:
            query: User's natural language question.
            top_k: Number of results to return.
            doc_id: Optional filter for a specific document.

        Returns:
            List of dicts with keys: text, doc_id, filename, chunk_index, score
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K

        # Embed the query
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        # Build query parameters
        where_filter = {"doc_id": doc_id} if doc_id else None

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[Retriever] Search error: {e}")
            return []

        # Parse results
        chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score: 1 - (distance / 2)
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2)

                chunks.append({
                    "text": results["documents"][0][i],
                    "doc_id": results["metadatas"][0][i]["doc_id"],
                    "filename": results["metadatas"][0][i]["filename"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "ai_probability": results["metadatas"][0][i].get("ai_probability", 0.0),
                    "source_url": results["metadatas"][0][i].get("source_url", ""),
                    "score": round(similarity, 4),
                })

        return chunks

    def get_chunks_with_sources(
        self,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Directly fetch chunks that have an internet source_url.
        """
        where_filter = {"source_url": {"$ne": ""}}
        if doc_id:
            where_filter = {"$and": [{"doc_id": {"$eq": doc_id}}, {"source_url": {"$ne": ""}}]}
            
        try:
            results = self.collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
                limit=top_k
            )
        except Exception as e:
            print(f"[Retriever] get_chunks_with_sources error: {e}")
            return []
            
        chunks = []
        if results and results.get("documents"):
            for i in range(len(results["documents"])):
                chunks.append({
                    "text": results["documents"][i],
                    "doc_id": results["metadatas"][i]["doc_id"],
                    "filename": results["metadatas"][i]["filename"],
                    "chunk_index": results["metadatas"][i]["chunk_index"],
                    "ai_probability": results["metadatas"][i].get("ai_probability", 0.0),
                    "source_url": results["metadatas"][i].get("source_url", ""),
                    "score": 1.0,  # Exact metadata match
                })
        return chunks

    def delete_document(self, doc_id: str):
        """Remove all chunks for a document from ChromaDB."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            print(f"[Retriever] Delete error: {e}")

    def is_healthy(self) -> bool:
        """Check if ChromaDB is reachable."""
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception:
            return False
