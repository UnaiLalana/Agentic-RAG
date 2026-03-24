# 📚 Document Retrieval-Augmented Generation (RAG) System

A complete local document intelligence system based on Retrieval-Augmented Generation. Upload PDF and Word documents, then ask questions in natural language and receive grounded answers with source citations.

**All components run locally via Docker — no external APIs, no cloud costs.**

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│  Flask API   │────▶│   MinIO      │
│   Frontend   │     │  (Port 5000) │     │  (Port 9000) │
│  (Port 8501) │     │              │     │  Object Store│
└──────────────┘     │  ┌────────┐  │     └──────────────┘
                     │  │ Parse  │  │
                     │  │ Chunk  │  │     ┌──────────────┐
                     │  │Retrieve│──│────▶│   ChromaDB   │
                     │  │Generate│  │     │  (Port 8000) │
                     │  └────────┘  │     │ Vector Search│
                     │              │     └──────────────┘
                     │              │
                     │              │     ┌──────────────┐
                     │              │────▶│  llama.cpp   │
                     │              │     │  (Port 8080) │
                     └──────────────┘     │ Quantized LLM│
                                          └──────────────┘
```

### Components

| Service | Technology | Port | Purpose |
|---------|-----------|------|---------|
| **Flask API** | Flask + Gunicorn | 5000 | Central orchestration (ingest, retrieve, generate) |
| **MinIO** | MinIO S3 | 9000/9001 | Raw document storage (PDF, DOCX) |
| **ChromaDB** | ChromaDB | 8000 | Vector index for chunk retrieval |
| **LLM Server** | llama.cpp | 8080 | Local quantized language model |
| **Frontend** | Streamlit | 8501 | Web UI (bonus) |

### Pipeline

1. **Ingest** — Upload PDF/DOCX → store in MinIO → extract text
2. **Chunk** — Recursive splitting (paragraphs → lines → sentences) with overlap
3. **Index** — Embed chunks with `all-MiniLM-L6-v2` → store in ChromaDB
4. **Retrieve** — Bi-encoder cosine similarity search → top-k chunks
5. **Generate** — Assemble grounded prompt → call llama.cpp → cited answer

---

## 📋 Prerequisites

- **Docker** and **Docker Compose** installed
- ~10 GB disk space (model + containers)
- Internet connection (first run only — to pull images)

---

## 📥 Model Download

> ⚠️ **Do NOT commit model weights to Git.** The GGUF file is ~4.4 GB.

Download the quantized Mistral model from [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF):

**Linux / macOS (bash):**
```bash
mkdir -p models

# Option 1: Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  --include "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" --local-dir models/

# Option 2: Using curl
curl -L -o models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

**Windows (PowerShell):**
```powershell
# Option 1: Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF --include "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" --local-dir models/

# Option 2: Using Invoke-WebRequest
Invoke-WebRequest -Uri "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" -OutFile "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
```

> **Expected file**: `models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (~4.4 GB)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd assignment3

# 2. Download the model (see above)

# 3. Start all services
docker-compose up --build

# 4. Wait for all services to be healthy (~1-2 minutes)
# The Flask API will start once MinIO, ChromaDB, and the LLM are ready

# 5. Verify
curl http://localhost:5000/health
```

**Access points:**
- Flask API: http://localhost:5000
- Streamlit UI: http://localhost:8501
- MinIO Console: http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`)

---

## 📡 API Reference

### `GET /health` — Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "minio": "ok",
  "chromadb": "ok",
  "llm": "ok",
  "status": "healthy"
}
```

### `POST /documents` — Upload a Document

```bash
curl -X POST -F "file=@report.pdf" http://localhost:5000/documents
```

**Response (201):**
```json
{
  "id": "a1b2c3d4-...",
  "filename": "report.pdf",
  "upload_date": "2026-03-11T...",
  "chunk_count": 42,
  "file_size": 156789,
  "message": "Document uploaded and indexed successfully (42 chunks)."
}
```

### `GET /documents` — List Documents

```bash
curl http://localhost:5000/documents
```

**Response (200):**
```json
{
  "documents": [
    {"id": "a1b2c3d4-...", "filename": "report.pdf", "upload_date": "...", "chunk_count": 42, "file_size": 156789}
  ],
  "total": 1
}
```

### `DELETE /documents/{id}` — Delete a Document

```bash
curl -X DELETE http://localhost:5000/documents/a1b2c3d4-...
```

**Response (200):**
```json
{"message": "Document 'report.pdf' (a1b2c3d4-...) deleted successfully."}
```

### `POST /query` — Ask a Question

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion of the report?", "top_k": 5}'
```

**Response (200):**
```json
{
  "answer": "Based on [1] and [2], the main conclusion is...",
  "question": "What is the main conclusion of the report?",
  "sources": [
    {
      "text": "The study concludes that...",
      "doc_id": "a1b2c3d4-...",
      "filename": "report.pdf",
      "chunk_index": 15,
      "relevance_score": 0.8734
    }
  ],
  "num_sources": 5
}
```

---

## 📊 Evaluation

Run the retrieval evaluation:

```bash
# 1. First, upload the evaluation documents to the system

# 2. Run the evaluation script
python eval/evaluate.py --api-url http://localhost:5000

# Results are saved to results/eval_results.json
```

**Metrics computed:**
- **Hit Rate @k** — fraction of queries where a relevant chunk is in top-k
- **MRR** — Mean Reciprocal Rank of the first relevant result
- **Precision @k** — fraction of top-k results that are relevant

---

## 🛠️ Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Recursive chunking** | Preserves document structure (paragraphs, sentences) better than fixed-size windows |
| **all-MiniLM-L6-v2** | Fast, CPU-friendly, 384-dim vectors; well-supported by ChromaDB |
| **Mistral-7B Q4_K_M** | Good instruction-following quality at ~4.4 GB; runs on CPU via llama.cpp |
| **SQLite metadata** | Lightweight document registry; avoids over-engineering for this use case |
| **Gunicorn** | Production-grade WSGI server; handles concurrent requests |

---

## ⚠️ Known Limitations

1. **No OCR** — Scanned/image-only PDFs are rejected with a clear error (no OCR capability)
2. **CPU inference** — LLM generation is slow on CPU (~30-60s per query); use a GPU for faster responses
3. **Single collection** — All documents share one ChromaDB collection; very large corpora may need sharding
4. **No authentication** — The API has no auth layer; intended for local development only
5. **Context window** — Mistral-7B has a 4096 token context; very long answers may be truncated
6. **Evaluation dataset** — Uses placeholder questions; replace with your own documents and questions for meaningful metrics

---

## 📁 Project Structure

```
assignment3/
├── docker-compose.yml          # Orchestrates all services
├── app/
│   ├── Dockerfile              # Flask app container
│   ├── requirements.txt        # Python dependencies
│   ├── main.py                 # Flask entry point
│   ├── config.py               # Environment config
│   ├── routes/
│   │   ├── health.py           # GET /health
│   │   ├── documents.py        # POST/GET/DELETE /documents
│   │   └── query.py            # POST /query
│   └── services/
│       ├── minio_service.py    # MinIO file operations
│       ├── parser.py           # PDF/DOCX text extraction
│       ├── chunker.py          # Recursive text chunking
│       ├── retriever.py        # ChromaDB + embeddings
│       └── llm_service.py      # llama.cpp API calls
├── frontend/
│   ├── Dockerfile              # Streamlit container
│   └── streamlit_app.py        # Web UI
├── eval/
│   ├── eval_dataset.json       # Annotated evaluation questions
│   └── evaluate.py             # Metrics computation
├── results/                    # Evaluation output
├── report/                     # Technical report (PDF)
├── models/                     # GGUF model file (gitignored)
└── README.md                   # This file
```
