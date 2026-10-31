"""
DocuMind – AI Powered PDF Chatbot
==================================
Main Streamlit application interface for DocuMind.
Provides an interactive, modern user interface for uploading PDF documents,
generating FAISS vector embeddings, and chatting with documents via RAG powered by Google Gemini.
"""

import os
import streamlit as st
from typing import List

import config
from src.pdf_processor import PDFProcessor, PDFProcessingError
from src.embeddings import get_embedding_model
from src.vector_store import VectorStoreManager, VectorStoreError
from src.rag_pipeline import RAGPipeline


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, portfolio-quality UI
st.markdown("""
<style>
    /* Global Typography & Spacing */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Status Badges */
    .badge-success {
        display: inline-block;
        padding: 4px 10px;
        background-color: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-info {
        display: inline-block;
        padding: 4px 10px;
        background-color: #EFF6FF;
        color: #1E40AF;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Info Cards */
    .feature-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease-in-out;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .feature-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #0F172A;
        margin-bottom: 0.4rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Citation Box */
    .citation-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .citation-header {
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .citation-snippet {
        color: #475569;
        font-style: italic;
        font-family: inherit;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = {}


# ==============================================================================
# Helper Functions
# ==============================================================================
def reset_conversation():
    """Clear conversation history while retaining loaded documents."""
    st.session_state.messages = []


def reset_all():
    """Clear conversation history and unload processed documents."""
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.processed_files = []
    st.session_state.doc_summary = {}


# ==============================================================================
# Sidebar UI: Configuration & Document Upload
# ==============================================================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/8297/8297437.png",
        width=56
    )
    st.title("DocuMind")
    st.caption("AI-Powered PDF Assistant • RAG Pipeline")
    st.divider()

    # 1. API Key Configuration
    st.subheader("🔑 API Key Setup")
    env_api_key = config.get_google_api_key()

    if env_api_key:
        st.markdown(
            '<div class="badge-success">✓ Loaded from .env</div>',
            unsafe_allow_html=True
        )
        api_key_input = st.text_input(
            "Google Gemini API Key (Optional Override):",
            value=env_api_key,
            type="password",
            help="Preloaded from environment. You can override it here if desired."
        )
    else:
        st.markdown(
            '<div class="badge-info">ℹ Enter Gemini API Key below</div>',
            unsafe_allow_html=True
        )
        api_key_input = st.text_input(
            "Google Gemini API Key:",
            type="password",
            placeholder="AIzaSy...",
            help="Get your free API key at https://aistudio.google.com/app/apikey"
        )

    active_api_key = api_key_input.strip() or env_api_key

    st.divider()

    # 2. PDF Upload Section
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF file(s):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents for question answering."
    )

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        process_button = st.button("⚡ Process PDFs", type="primary", use_container_width=True)

    with col_btn2:
        clear_button = st.button("🗑️ Reset All", on_click=reset_all, use_container_width=True)

    # Process PDFs Action
    if process_button:
        if not uploaded_files:
            st.error("Please upload at least one PDF file first.")
        else:
            with st.spinner("Extracting text and chunking documents..."):
                processor = PDFProcessor(
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP
                )
                try:
                    chunked_docs, summary = processor.process_multiple_pdfs(uploaded_files)

                    if not chunked_docs:
                        st.error("No valid text could be extracted from the uploaded PDF(s).")
                    else:
                        st.spinner("Generating HuggingFace embeddings and indexing with FAISS...")
                        embeddings = get_embedding_model()
                        vector_store = VectorStoreManager.create_vector_store(chunked_docs, embeddings)

                        # Update session state
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.doc_summary = summary

                        st.success(f"Successfully processed {summary['processed_files']} file(s) into {summary['total_chunks']} chunks!")

                except PDFProcessingError as pe:
                    st.error(f"PDF Processing Error: {str(pe)}")
                except VectorStoreError as ve:
                    st.error(f"Vector Store Error: {str(ve)}")
                except Exception as ex:
                    st.error(f"An unexpected error occurred: {str(ex)}")

    # Document Processing Status
    if st.session_state.vector_store is not None:
        st.divider()
        st.subheader("📊 Document Status")
        st.markdown(f"**Loaded Files:** {len(st.session_state.processed_files)}")
        for fn in st.session_state.processed_files:
            st.caption(f"• 📄 {fn}")

        summary = st.session_state.doc_summary
        if summary:
            st.markdown(f"**Total Pages:** {summary.get('total_pages', 'N/A')}")
            st.markdown(f"**Indexed Chunks:** {summary.get('total_chunks', 'N/A')}")

    # Advanced Settings Expander
    st.divider()
    with st.expander("⚙️ Advanced Parameters"):
        selected_model = st.selectbox(
            "Gemini Model:",
            options=config.AVAILABLE_LLM_MODELS,
            index=config.AVAILABLE_LLM_MODELS.index(config.DEFAULT_LLM_MODEL) if config.DEFAULT_LLM_MODEL in config.AVAILABLE_LLM_MODELS else 0
        )
        selected_temp = st.slider(
            "Temperature (Creativity):",
            min_value=0.0,
            max_value=1.0,
            value=config.DEFAULT_TEMPERATURE,
            step=0.05,
            help="Lower values produce factual, document-grounded responses."
        )
        top_k_chunks = st.slider(
            "Top-K Chunks to Retrieve:",
            min_value=2,
            max_value=8,
            value=config.TOP_K_RESULTS,
            step=1,
            help="Number of most similar text chunks passed to the LLM."
        )


# ==============================================================================
# Main Content Area
# ==============================================================================

# Header Title & Description
st.markdown('<div class="main-header">DocuMind – AI PDF Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask accurate, contextual questions about your documents powered by Retrieval-Augmented Generation (RAG).</div>',
    unsafe_allow_html=True
)

# Top Bar Action: Clear Chat button
if st.session_state.messages:
    col_chat_stat, col_chat_clear = st.columns([6, 1])
    with col_chat_stat:
        st.caption(f"Active Conversation: {len(st.session_state.messages) // 2} turn(s)")
    with col_chat_clear:
        st.button("Clear Chat", on_click=reset_conversation, use_container_width=True)

# ------------------------------------------------------------------------------
# Welcome & Guide View (When No Documents or Chat yet)
# ------------------------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("👋 **Welcome to DocuMind!** Get started by uploading a PDF document in the left sidebar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">1. Upload PDF</div>
            <div class="feature-desc">Upload one or multiple PDF documents. DocuMind extracts page-level text and splits it into semantic chunks.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">2. FAISS Vector Search</div>
            <div class="feature-desc">Generates 384-dimensional dense vectors using HuggingFace all-MiniLM-L6-v2 embeddings stored in FAISS.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">3. Grounded Gemini AI</div>
            <div class="feature-desc">Retrieves relevant context and passes it to Google Gemini with source document and page number citations.</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Chat History Display
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display source citations if present
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 Source Citations ({len(sources)} referenced sections)"):
                for src in sources:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">📄 {src.get('source', 'Document')} — Page {src.get('page', 'N/A')}</div>
                        <div class="citation-snippet">"{src.get('snippet', '')}"</div>
                    </div>
                    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Chat Input & RAG Execution
# ------------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about your uploaded document(s)...")

if user_query:
    # 1. Validate vector store
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF document in the sidebar before asking questions.")
    # 2. Validate API Key
    elif not active_api_key:
        st.error("⚠️ Please provide a Google Gemini API Key in the sidebar or via the .env file.")
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("DocuMind is retrieving document context and generating answer..."):
                pipeline = RAGPipeline(
                    api_key=active_api_key,
                    model_name=selected_model,
                    temperature=selected_temp,
                    top_k=top_k_chunks
                )

                result = pipeline.answer_question(
                    query=user_query,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.messages[:-1]
                )

                answer_text = result["answer"]
                sources = result["sources"]

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Source Citations ({len(sources)} referenced sections)"):
                        for src in sources:
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">📄 {src.get('source', 'Document')} — Page {src.get('page', 'N/A')}</div>
                                <div class="citation-snippet">"{src.get('snippet', '')}"</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Save assistant message and sources to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources
        })
