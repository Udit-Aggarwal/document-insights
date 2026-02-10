
# app.py
"""
Streamlit Document Summarization & Q&A (Transformers pipelines, open-source)

- Summarizer: sshleifer/distilbart-cnn-12-6   (lightweight BART variant for summarization)
- Q&A generator: google/flan-t5-base          (instruction-tuned seq2seq)

Install (CPU):
    pip install -U streamlit PyPDF2 faiss-cpu langchain-text-splitters langchain-community \
                   langchain-huggingface transformers sentencepiece

Run:
    streamlit run app.py
"""

import gc
import os
import streamlit as st
from typing import Dict, Any
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# We keep torch optional; if unavailable, pipelines will run on CPU.
try:
    import torch
except Exception:
    torch = None

from transformers import pipeline

# ---------------------------
# Page & session setup
# ---------------------------
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A")

for k, v in {
    "vectorstore": None,
    "text": "",
    "chunks": [],
    "debug_raw": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 10, 4, 1)
st.sidebar.markdown("---")
st.session_state.debug_raw = st.sidebar.checkbox("Show raw model output", value=False)

# ---------------------------
# Fixed open-source models
# ---------------------------
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "google/flan-t5-base"

# Device note (informative only)
if torch and torch.cuda.is_available():
    device_note = "CUDA GPU"
elif torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_note = "Apple MPS (CPU fallback in pipelines)"
else:
    device_note = "CPU"
st.sidebar.caption(f"Compute device: **{device_note}**")

# ---------------------------
# Upload area
# ---------------------------
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
            st.session_state.text = "\n\n".join(pages)
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

# ---------------------------
# Process document (split + embed + index)
# ---------------------------
if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.success(f"Document processed into {len(chunks)} chunks and indexed.")
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")

if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def _cuda_kwargs() -> Dict[str, Any]:
    """Pass device only when CUDA is available; otherwise rely on CPU."""
    if torch and torch.cuda.is_available():
        return {"device": 0}
    return {}

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # DistilBART summarization pipeline
    return pipeline("summarization", model=SUMMARIZER_MODEL, **_cuda_kwargs())

@st.cache_resource(show_spinner=False)
def load_qa_generator():
    # FLAN-T5 text2text pipeline
    return pipeline("text2text-generation", model=QA_MODEL, **_cuda_kwargs())

summarizer = load_summarizer()
qa_gen = load_qa_generator()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📝 Summary", "❓ Q&A"])

# ---------------------------
# Summary tab
# ---------------------------
with tab1:
    st.markdown("**Generate a short summary of the document.**")
    summary_max_input = st.slider(
        "Max input characters to summarize (trim long docs)",
        min_value=256, max_value=8192, value=2048, step=256
    )
    target_words = st.slider("Target summary length (words)", 50, 400, 120, 10)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            try:
                input_text = st.session_state.text[:summary_max_input]

                # Convert desired words to a rough token budget.
                approx_tokens = int(target_words * 1.3)

                outputs = summarizer(
                    input_text,
                    max_new_tokens=approx_tokens,
                    do_sample=False,
                    num_beams=4,
                    truncation=True,
                )
                # Pipeline returns list[dict] with 'summary_text'
                summary = outputs[0].get("summary_text", "").strip() if outputs else ""

                if st.session_state.debug_raw:
                    st.subheader("Raw model output")
                    st.write(outputs)

                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.warning("No summary text returned by the model.")
            except Exception as e:
                st.error(f"Summarization failed: {e}")
            finally:
                gc.collect()

# ---------------------------
# Q&A tab
# ---------------------------
with tab2:
    st.markdown("**Ask a question about the document.**")
    question = st.text_input("Question", "")
    max_tokens_ans = st.slider("Max answer tokens", 64, 512, 192, 16)

    if st.button("Get Answer") and question.strip():
        with st.spinner("Retrieving and generating answer..."):
            try:
                # 1) Retrieve top-k relevant chunks
                docs = st.session_state.vectorstore.similarity_search(question, k=int(top_k))
                if not docs:
                    st.warning("No relevant documents found.")
                    st.stop()

                # Keep context length reasonable
                # (DistilBART/FLAN-T5 handle ~512-1024 tokens comfortably)
                context = "\n\n".join([d.page_content for d in docs])[:6000]

                # 2) Prompt for FLAN-T5
                prompt = (
                    "You are a helpful assistant. Use ONLY the context to answer the question.\n"
                    "If the answer is not in the context, say \"I don't know based on the provided context.\" \n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer concisely in 2-5 sentences."
                )

                outputs = qa_gen(
                    prompt,
                    max_new_tokens=max_tokens_ans,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=4,
                )
                answer = outputs[0].get("generated_text", "").strip() if outputs else ""

                if st.session_state.debug_raw:
                    st.subheader("Raw model output")
                    st.write(outputs)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Retrieved snippets"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**Snippet {i}**")
                        st.write(d.page_content[:1000])
            except Exception as e:
                st.error(f"Q&A failed: {e}")
            finally:
                gc.collect()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("This app uses FAISS for vector search, MiniLM embeddings, and Transformers pipelines for generation.")
