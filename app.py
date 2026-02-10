
 app.py
"""
Streamlit Document Summarization & Q&A

Requirements:
    pip install -U streamlit PyPDF2 langchain-text-splitters langchain-huggingface langchain-community faiss-cpu

Environment:
    Export a Hugging Face token with access to the chosen models:
      - macOS/Linux:   export HUGGINGFACE_API_TOKEN=hf_************************
      - Windows PS:    $env:HUGGINGFACE_API_TOKEN="hf_************************"

Notes:
    - This app uses FAISS for local vector search and Hugging Face Inference via LangChain's HuggingFaceEndpoint.
    - IMPORTANT: HuggingFaceEndpoint.invoke() expects a STRING (or messages), not a dict.
    - IMPORTANT: Use 'max_new_tokens' (not 'max_length') for InferenceClient text generation params.
"""

import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_community.vectorstores import FAISS

# =========================
# Page & session setup
# =========================
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input(
    "Chunk size", min_value=200, max_value=4000, value=1000, step=100
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap", min_value=0, max_value=1000, value=200, step=50
)
top_k = st.sidebar.number_input(
    "Top K documents for retrieval", min_value=1, max_value=10, value=4, step=1
)
st.sidebar.markdown("---")
st.session_state.debug_raw = st.sidebar.checkbox("Show raw LLM response", value=False)

# =========================
# Upload area
# =========================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
            st.session_state.text = "\n\n".join(pages)
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8")
        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

# =========================
# Process document
# =========================
if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success(
                f"Document processed into {len(chunks)} chunks and indexed."
            )
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")

# Require vectorstore for downstream
if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()

# =========================
# Hugging Face token
# =========================
api_token = os.getenv("HUGGINGFACE_API_TOKEN") or st.secrets.get(
    "HUGGINGFACE_API_TOKEN", None
)
if not api_token:
    st.error("⚠️ HUGGINGFACE_API_TOKEN not found. Set it in environment or Streamlit secrets.")
    st.stop()

# =========================
# Model configs (open-source)
# =========================
# Primary summarizer (lighter, open-source alternative to facebook/bart-large-cnn)
SUMMARIZER_REPO = "sshleifer/distilbart-cnn-12-6"
# Other good options:
#   "facebook/bart-large-cnn"
#   "google/pegasus-cnn_dailymail"
#   "google/pegasus-xsum"

# Fallback instruction model (open-source)
FALLBACK_REPO = "google/flan-t5-base"   # use "-large" if you have more resources

# Q&A generation model (open-source)
QA_REPO = "google/flan-t5-base"

# =========================
# Helper to normalize LLM responses
# =========================
def normalize_llm_response(raw):
    """
    Normalize common return shapes to a string.
    With LangChain's HuggingFaceEndpoint, you typically get a string already.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("summary_text", "generated_text", "output", "text", "result"):
            if key in raw and raw[key]:
                return raw[key]
        if "error" in raw:
            return f"ERROR: {raw['error']}"
        return str(raw)
    if isinstance(raw, (list, tuple)):
        try:
            return "\n".join([normalize_llm_response(r) for r in raw])
        except Exception:
            return str(raw)
    return str(raw)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])

# =========================
# Summary tab
# =========================
with tab1:
    st.markdown("**Generate a short summary of the document.**")
    summary_max_input = st.slider(
        "Max input characters to summarize (trim long docs)", 256, 8192, 2048, step=256
    )
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            input_text = st.session_state.text[:summary_max_input]

            try:
                # --- Primary summarizer (string input; params via model_kwargs)
                summarizer = HuggingFaceEndpoint(
                    repo_id=SUMMARIZER_REPO,
                    huggingfacehub_api_token=api_token,
                    task="summarization",
                    model_kwargs={
                        # Use max_new_tokens instead of max_length for InferenceClient
                        "max_new_tokens": 180,
                        "do_sample": False,
                        "return_full_text": False
                    },
                )
                raw = summarizer.invoke(input_text)  # <-- PASS STRING, NOT DICT
            except Exception:
                # --- Fallback to instruction model (string input)
                try:
                    fallback_llm = HuggingFaceEndpoint(
                        repo_id=FALLBACK_REPO,
                        huggingfacehub_api_token=api_token,
                        model_kwargs={
                            "max_new_tokens": 180,
                            "do_sample": False,
                            "return_full_text": False
                        },
                    )
                    prompt = "Summarize the following text in 3 sentences:\n\n" + input_text
                    raw = fallback_llm.invoke(prompt)  # <-- PASS STRING, NOT DICT
                except Exception as e2:
                    st.error(f"Summarization failed: {e2}")
                    raw = None

            if st.session_state.debug_raw:
                st.subheader("Raw LLM Response")
                st.write(raw)

            summary = normalize_llm_response(raw)
            if summary:
                st.subheader("Summary")
                st.write(summary)
            else:
                st.warning("No summary returned. Toggle debug to inspect raw response.")

# =========================
# Q&A tab
# =========================
with tab2:
    st.markdown("**Ask a question about the document.**")
    question = st.text_input("Question", "")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Finding answer..."):
            try:
                # 1) Retrieve top-k relevant chunks
                docs = st.session_state.vectorstore.similarity_search(question, k=int(top_k))
                if not docs:
                    st.warning("No relevant documents found.")
                    st.stop()

                context = "\n\n".join([d.page_content for d in docs])

                # 2) Build prompt
                prompt_template = (
                    "You are an assistant that answers questions using the provided context.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer concisely and cite context when helpful."
                )
                prompt_text = prompt_template.format(context=context, question=question)

                # 3) Call the LLM endpoint (string input; params via model_kwargs)
                qa_llm = HuggingFaceEndpoint(
                    repo_id=QA_REPO,
                    huggingfacehub_api_token=api_token,
                    model_kwargs={
                        "max_new_tokens": 256,
                        "do_sample": False,
                        "return_full_text": False
                    },
                )

                answer_raw = qa_llm.invoke(prompt_text)

                if st.session_state.debug_raw:
                    st.subheader("Raw LLM Response")
                    st.write(answer_raw)

                answer = normalize_llm_response(answer_raw)

                st.subheader("Answer")
                st.write(answer)

                # Show retrieved snippets for transparency
                with st.expander("Retrieved snippets"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Snippet {i}**")
                        st.write(d.page_content[:1000])

            except Exception as e:
                st.error(f"Q&A failed: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Notes • This app uses FAISS for vector search and Hugging Face endpoints for LLMs and embeddings."
)
