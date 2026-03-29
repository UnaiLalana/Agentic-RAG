# 📚 Document Integrity Analysis System

A production-ready system for **AI-generated text detection**, **internet plagiarism source attribution**, and **intelligent document Q&A**, powered by a fine-tuned BERT classifier, web search, and Retrieval-Augmented Generation (RAG).

Upload PDF or DOCX documents ➜ each paragraph is classified as human-written or AI-generated ➜ human-written paragraphs are checked against the internet for copy-paste plagiarism ➜ ask natural language questions about the content and get grounded, cited answers.

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
                     │  │AI Detect│──│────▶│   ChromaDB   │
                     │  │Src Search│ │     │  (Port 8000) │
                     │  │Retrieve│  │     │ Vector Search│
                     │  │Generate│  │     └──────────────┘
                     │  └────────┘  │
                     │              │     ┌──────────────┐
                     │              │────▶│  llama.cpp   │
                     │              │     │  (Port 8080) │
                     └──────────────┘     │ Quantized LLM│
                                          └──────────────┘
```

### Components

| Service | Technology | Port | Purpose |
|---------|-----------|------|---------|
| **Flask API** | Flask + Gunicorn | 5000 | Central orchestration (ingest, AI detection, source search, retrieve, generate) |
| **MinIO** | MinIO S3 | 9000/9001 | Raw document storage (PDF, DOCX) |
| **ChromaDB** | ChromaDB | 8000 | Vector index for chunk retrieval with metadata (AI probability, source URL) |
| **LLM Server** | llama.cpp | 8080 | Local quantized language model (Mistral-7B) |
| **Frontend** | Streamlit | 8501 | Web UI for upload, AI analysis visualization, and Q&A |

### Pipeline

1. **Ingest** — Upload PDF/DOCX → store in MinIO → extract text
2. **Chunk** — Semantic paragraph splitting (double newlines → sentence boundaries) with PDF reflow
3. **AI Detection** — Each chunk is classified by a fine-tuned `bert-base-uncased` model → AI probability score
4. **Source Search** — Chunks with low AI probability are searched on the web via DuckDuckGo → source URL attribution
5. **Index** — Embed chunks with `all-MiniLM-L6-v2` → store in ChromaDB with metadata (AI probability, source URL)
6. **Retrieve** — Bi-encoder cosine similarity search → top-k chunks (with intent routing for plagiarism queries)
7. **Generate** — Assemble grounded prompt → call Mistral-7B via llama.cpp → cited answer

---

## 🤖 AI Detection Model

The AI detection component uses a **fine-tuned `bert-base-uncased`** (110M parameters) with mean pooling, trained on the [PAN 2025 Voight-Kampff](https://pan.webis.de/clef25/pan25-web/generative-ai-authorship-verification.html) dataset for binary classification of AI-generated vs. human-written academic text.

### Results

| Metric | Value |
|--------|-------|
| Validation Composite | 0.8608 |
| **Test Composite** | **0.8534** |
| Test F1-macro | 0.8303 |
| Test ROC-AUC | 0.8824 |
| Best Pooling | Mean |
| Training Time | ~3.2 hours (GPU) |

The composite metric follows the PAN 2025 scoring scheme:
`Score = 1/5 (AUC + (1−Brier) + F1 + F0.5u + c@1)`

### Training

See [`ai_detection/train_best_bert.py`](ai_detection/train_best_bert.py):
- 80/20 stratified train-test split, then 85/15 train-validation split
- AdamW (lr=2e-5, weight_decay=0.01), linear warmup (10%)
- Up to 8 epochs with early stopping (patience 3)
- Gradient clipping (max norm 1.0), batch size 16, max_length 512
- All random seeds set to 42 (random, numpy, torch, cuda)
- Ablation: CLS pooling (0.8502) vs Mean pooling (**0.8608**) → Mean wins

---

## 📋 Prerequisites

- **Docker** and **Docker Compose** installed
- ~10 GB disk space (model + containers)
- Internet connection (first run only — to pull images)

---

## 📥 Model Download

> ⚠️ **Do NOT commit model weights to Git.** The GGUF file is ~4.4 GB and the BERT model is ~440 MB.

### Mistral-7B (LLM for Q&A)

Download the quantized Mistral model from [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF):

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

### BERT AI Detector

Train the model locally (requires GPU recommended):

```bash
cd ai_detection
python train_best_bert.py
# Saves to: results/bert_best_model.pt
# Copy to models/ for deployment:
cp ../results/bert_best_model.pt ../models/ai_detector.pt
```

Or download a pre-trained checkpoint (if available from the team).

> **Expected files:**
> - `models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (~4.4 GB)
> - `models/ai_detector.pt` (~440 MB)
> - `models/pretrained_transformer.py` (model class definition)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd Agentic-RAG

# 2. Download/train the models (see above)

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

Uploads a PDF/DOCX, parses it, chunks it into paragraphs, classifies each chunk with the BERT AI detector, searches the web for source URLs on human-written chunks, and indexes everything in ChromaDB.

```bash
curl -X POST -F "file=@report.pdf" http://localhost:5000/documents
```

**Response (201):**
```json
{
  "id": "a1b2c3d4-...",
  "filename": "report.pdf",
  "upload_date": "2026-03-29T...",
  "chunk_count": 42,
  "file_size": 156789,
  "message": "Document uploaded and indexed successfully (42 chunks)."
}
```

### `GET /documents` — List Documents

```bash
curl http://localhost:5000/documents
```

### `GET /documents/{id}/chunks` — View AI Analysis

Returns all chunks for a document with their AI probability scores and detected source URLs.

```bash
curl http://localhost:5000/documents/a1b2c3d4-.../chunks
```

**Response (200):**
```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "text": "This paragraph was written by...",
      "ai_probability": 0.12,
      "source_url": ""
    },
    {
      "chunk_index": 1,
      "text": "Natural language processing is...",
      "ai_probability": 0.03,
      "source_url": "https://en.wikipedia.org/wiki/Natural_language_processing"
    }
  ]
}
```

### `DELETE /documents/{id}` — Delete a Document

```bash
curl -X DELETE http://localhost:5000/documents/a1b2c3d4-...
```

### `POST /query` — Ask a Question

Supports two modes via intent routing:
- **General Q&A:** Semantic retrieval → LLM-generated answer with citations
- **Plagiarism queries:** Keywords like "copy", "plagiarism", "source" trigger direct lookup of chunks with internet source URLs

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
      "relevance_score": 0.8734,
      "web_source": "https://example.com/original-source"
    }
  ],
  "num_sources": 5
}
```

---

## 📊 Evaluation

### RAG Retrieval Metrics

Evaluated on 10 hand-crafted questions spanning 2 documents:

```bash
python eval/evaluate.py --api-url http://localhost:5000
# Results saved to results/eval_results.json
```

| Metric | k=1 | k=3 | k=5 |
|--------|-----|-----|-----|
| **Hit Rate** | 0.900 | **1.000** | **1.000** |
| **MRR** | 0.900 | 0.933 | 0.933 |
| **Precision@k** | 0.900 | 0.733 | 0.720 |

### AI Detection Metrics

See the [AI Detection Model](#-ai-detection-model) section above for BERT results on the PAN 2025 dataset. Detailed training summary in [`results/bert_best_summary.csv`](results/bert_best_summary.csv).

---

## 🛠️ Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Paragraph chunking** | Preserves document structure; semantic units better than fixed-size windows for AI detection |
| **Mean pooling (BERT)** | Ablation showed +1.06 composite points over CLS pooling on academic text |
| **all-MiniLM-L6-v2** | Fast, CPU-friendly, 384-dim vectors; well-supported by ChromaDB |
| **Mistral-7B Q4_K_M** | Good instruction-following quality at ~4.4 GB; runs on CPU via llama.cpp |
| **DuckDuckGo multi-query** | 5 query variants per sentence with n-gram overlap scoring reduces false positives |
| **Intent routing** | Plagiarism queries bypass semantic search and directly fetch chunks with source URLs |
| **SQLite metadata** | Lightweight document registry; avoids over-engineering for this use case |
| **Docker Compose** | Reproducible, one-command deployment of all 5 services |

---

## ⚠️ Known Limitations

1. **AI Detection Ceiling** — Notation-heavy academic domains (math, physics) cause false negatives; stylistic detection fails when domain formalisms mask authorship signals
2. **CPU inference** — LLM generation is slow on CPU (~30-60s per query); use a GPU for faster responses
3. **No OCR** — Scanned/image-only PDFs are rejected (no OCR capability)
4. **Source search rate limits** — DuckDuckGo may throttle requests during batch document processing
5. **Context window** — Mistral-7B has a 4096 token context; very long answers may be truncated
6. **Single collection** — All documents share one ChromaDB collection; very large corpora may need sharding
7. **No authentication** — The API has no auth layer; intended for local development only
8. **Bias** — AI detector trained on English academic text only; performance on informal or multilingual text is unvalidated

---

## 📁 Project Structure

```
Agentic-RAG/
├── docker-compose.yml              # Orchestrates all 5 services
├── pyproject.toml                   # Python dependencies (uv)
├── README.md                        # This file
│
├── app/                             # Flask API backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                      # Flask entry point
│   ├── config.py                    # Environment config
│   ├── database.py                  # SQLite metadata store
│   ├── routes/
│   │   ├── health.py                # GET /health
│   │   ├── documents.py             # POST/GET/DELETE /documents, GET /documents/{id}/chunks
│   │   └── query.py                 # POST /query (intent routing + RAG)
│   └── services/
│       ├── ai_detector.py           # BERT-based AI detection singleton
│       ├── source_searcher.py       # DuckDuckGo web source attribution
│       ├── parser.py                # PDF/DOCX text extraction
│       ├── chunker.py               # Semantic paragraph chunking
│       ├── retriever.py             # ChromaDB + sentence-transformers retrieval
│       ├── llm_service.py           # Mistral-7B via llama.cpp
│       └── minio_service.py         # MinIO file operations
│
├── ai_detection/                    # BERT training scripts
│   └── train_best_bert.py           # Full training pipeline with ablation
│
├── models/                          # Model files (gitignored weights)
│   └── pretrained_transformer.py    # TransformerClassifier + BERTDataset classes
│
├── frontend/                        # Streamlit web UI
│   ├── Dockerfile
│   └── streamlit_app.py             # Upload, AI analysis view, Q&A interface
│
├── eval/                            # Evaluation
│   ├── eval_dataset.json            # 10 annotated evaluation questions
│   └── evaluate.py                  # Hit Rate, MRR, Precision@k computation
│
├── results/                         # Evaluation & training outputs
│   ├── eval_results.json            # RAG retrieval evaluation results
│   └── bert_best_summary.csv        # BERT training summary metrics
│
└── report/                          # Technical report (PDF)
    └── Capstone_Project.pdf         # ACL-format technical paper
```

---

## 📄 Technical Report

The full technical report is available at [`report/Capstone_Project.pdf`](report/Capstone_Project.pdf), written in ACL conference format. It covers:

- System architecture and design
- BERT fine-tuning methodology and ablation studies
- Non-deep-learning baseline comparison (TF-IDF + LR, Word2Vec + LR)
- RAG retrieval evaluation
- Error analysis of AI detection failures
- Discussion of limitations, bias, and ethical considerations
