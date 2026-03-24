import os
import streamlit as st
import requests
import time

# Flask API URL
API_URL = os.getenv("API_URL", "http://localhost:5000")

# ─────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG Document Intelligence",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Sidebar — System Health
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if st.button("🔄 Check Health"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            health = resp.json()
            for service, status in health.items():
                if service == "status":
                    continue
                icon = "✅" if status == "ok" else "❌"
                st.markdown(f"{icon} **{service.upper()}**: {status}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This **RAG Document Intelligence** system lets you upload "
        "PDF and DOCX documents, then ask questions in natural language. "
        "Answers are grounded in your documents with source citations."
    )

# ─────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────
st.markdown('<p class="main-header">📚 Document Intelligence System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload documents, ask questions, get grounded answers with citations.</p>',
    unsafe_allow_html=True,
)

tab_upload, tab_documents, tab_query = st.tabs(["📤 Upload", "📄 Documents", "💬 Ask a Question"])

# ─────────────────────────────────────────────────
# Tab 1: Upload Documents
# ─────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload a Document")
    st.markdown("Supported formats: **PDF**, **DOCX**")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx"],
        help="Upload a PDF or Word document to index for question answering.",
    )

    if uploaded_file is not None:
        if st.button("📤 Upload & Index", type="primary"):
            with st.spinner("Uploading, parsing, chunking, and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/documents", files=files, timeout=300)

                    if resp.status_code == 201:
                        result = resp.json()
                        st.success(
                            f"✅ **{result['filename']}** uploaded successfully!\n\n"
                            f"- **Document ID**: `{result['id']}`\n"
                            f"- **Chunks indexed**: {result['chunk_count']}\n"
                            f"- **File size**: {result['file_size']:,} bytes"
                        )
                    else:
                        error = resp.json().get("error", "Unknown error")
                        st.error(f"❌ Upload failed: {error}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────
# Tab 2: Document List
# ─────────────────────────────────────────────────
with tab_documents:
    st.markdown("### Uploaded Documents")

    if st.button("🔄 Refresh List"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/documents", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])

            if not docs:
                st.info("📭 No documents uploaded yet. Go to the Upload tab to add some!")
            else:
                st.markdown(f"**{len(docs)}** document(s) in the system:")

                for doc in docs:
                    with st.expander(f"📄 {doc['filename']} - {doc['chunk_count']} chunks"):
                        col1, col2, col3 = st.columns([4, 2, 2])
                        with col1:
                            st.caption(f"ID: `{doc['id']}`")
                        with col2:
                            st.caption(f"📅 {doc['upload_date'][:19]}")
                        with col3:
                            if st.button("🗑️ Delete", key=f"del_{doc['id']}", help="Delete this document"):
                                try:
                                    del_resp = requests.delete(
                                        f"{API_URL}/documents/{doc['id']}", timeout=30
                                    )
                                    if del_resp.status_code == 200:
                                        st.success("Deleted!")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete.")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                    
                        if st.button(f"🔍 View AI Analysis", key=f"ai_{doc['id']}"):
                            with st.spinner("Fetching document chunks..."):
                                try:
                                    ch_resp = requests.get(f"{API_URL}/documents/{doc['id']}/chunks", timeout=60)
                                    if ch_resp.status_code == 200:
                                        chunks = ch_resp.json().get("chunks", [])
                                        if not chunks:
                                            st.info("No chunks found for this document.")
                                        for c in chunks:
                                            prob = c.get('ai_probability', 0.0)
                                            url = c.get('source_url', '')
                                            bg_color = "#ffebee" if prob > 0.5 else "#e8f5e9"
                                            text_color = "#b71c1c" if prob > 0.5 else "#1b5e20"
                                            label = "🤖 AI-Generated" if prob > 0.5 else "👤 Human-Written"
                                            
                                            html = f'''
                                            <div style="background-color: {bg_color}; color: {text_color}; padding: 12px; margin: 8px 0; border-radius: 6px; border: 1px solid {text_color}40;">
                                                <div style="font-weight: bold; margin-bottom: 6px;">[{label} | Probability: {prob:.2%}]</div>
                                                <div style="line-height: 1.5;">{c.get('text', '')}</div>
                                            '''
                                            if url:
                                                html += f'<div style="margin-top: 10px; font-size: 0.9em;">🔗 <strong>Extracted Source:</strong> <a href="{url}" target="_blank" style="color: {text_color}; text-decoration: underline;">{url}</a></div>'
                                            html += "</div>"
                                            
                                            st.markdown(html, unsafe_allow_html=True)
                                    else:
                                        st.error("Could not fetch chunks.")
                                except Exception as e:
                                    st.error(f"Error fetching chunks: {e}")
        else:
            st.error("Could not fetch documents.")
    except Exception as e:
        st.warning(f"Cannot reach API: {e}")

# ─────────────────────────────────────────────────
# Tab 3: Query
# ─────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask a Question")
    st.markdown("Ask anything about your uploaded documents. Answers will cite source passages.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g., What is the main conclusion of the report?",
        height=100,
    )

    col_k, col_submit = st.columns([1, 3])
    with col_k:
        top_k = st.number_input("Sources (top_k)", min_value=1, max_value=20, value=5)
    with col_submit:
        submit = st.button("🔍 Search & Answer", type="primary")

    if submit and question.strip():
        with st.spinner("Searching documents and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip(), "top_k": top_k},
                    timeout=180,
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Answer
                    st.markdown("### 💡 Answer")
                    st.markdown(result["answer"])

                    # Sources
                    st.markdown("### 📎 Source Passages")
                    sources = result.get("sources", [])
                    if sources:
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-box">'
                                f'<strong>[{i}]</strong> '
                                f'<em>{src["filename"]}</em> '
                                f'(score: {src["relevance_score"]:.4f})<br/>'
                                f'{src["text"][:500]}{"..." if len(src["text"]) > 500 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No source passages found.")
                else:
                    error = resp.json().get("error", "Unknown error")
                    st.error(f"❌ Query failed: {error}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif submit:
        st.warning("Please enter a question.")
