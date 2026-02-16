import os
import io
import re
import uuid
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF

from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# ==========================
# Streamlit config
# ==========================
st.set_page_config(page_title="Doc Summarization & Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A (Stable: Process → Query)")

FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)
CHAT_DB_PATH = "chat_history.sqlite"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


# ==========================
# Helpers
# ==========================
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

def now_iso():
    return datetime.utcnow().isoformat()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize(s: str) -> str:
    return (s or "").strip()

def init_chat_db():
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def load_history(session_id: str) -> List[dict]:
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY id ASC", (session_id,))
    rows = cur.fetchall()
    con.close()
    return [{"role": r, "content": c} for (r, c) in rows]

def append_message(session_id: str, role: str, content: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now_iso())
    )
    con.commit()
    con.close()

def clear_history(session_id: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
    con.commit()
    con.close()

def save_faiss(vs: FAISS, path: Path = FAISS_DIR):
    vs.save_local(str(path))

def load_faiss(embeddings, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


# ==========================
# Cached resources
# ==========================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

@st.cache_resource
def get_qdrant_client():
    url = get_secret("QDRANT_URL", None)
    api_key = get_secret("QDRANT_API_KEY", None)
    if not url or not api_key:
        return None
    return QdrantClient(url=url, api_key=api_key)

def ensure_qdrant_collection(client: QdrantClient, collection_name: str):
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


# ==========================
# LLM routing (HF Router -> Groq, fallback Groq direct)
# ==========================
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

def llm_chat_with_fallback(model_id: str, messages: List[dict], temperature: float, max_tokens: int,
                          hf_token: Optional[str], groq_key: Optional[str], debug: bool=False) -> str:
    # Primary: HF Router -> Groq
    if hf_token:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            resp = hf_client.chat.completions.create(
                model=routed, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            if normalize(out):
                return normalize(out)
        except Exception as e:
            if debug:
                st.exception(e)

    # Fallback: Groq direct
    if groq_key:
        try:
            groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            resp = groq_client.chat.completions.create(
                model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            return normalize(out)
        except Exception as e:
            if debug:
                st.exception(e)

    return ""


# ==========================
# PDF extraction (page-aware)
# ==========================
def extract_pdf_page_texts(pdf_bytes: bytes) -> List[Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": txt})
    doc.close()
    return pages


# ==========================
# Session state
# ==========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id

init_chat_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None  # doc_id of current uploaded doc

if "doc_filename" not in st.session_state:
    st.session_state.doc_filename = None

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

if "page_texts" not in st.session_state:
    st.session_state.page_texts = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ==========================
# Sidebar
# ==========================
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local)"], index=0)

st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents", 1, 12, 6, 1)

st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 128, 4096, 1024, 64)
debug = st.sidebar.checkbox("Debug", value=False)

st.sidebar.header("Chat")
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    clear_history(SESSION_ID)
    st.toast("Chat cleared", icon="🧹")

st.sidebar.header("Sources (last answer)")
if st.session_state.last_sources:
    for s in st.session_state.last_sources:
        with st.sidebar.expander(f"{s['filename']} — Page {s['page']}"):
            st.write(s["preview"])


# ==========================
# Tokens
# ==========================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. HF Router primary route may fail.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. Groq fallback may fail.")


# ==========================
# Main UI: Upload + Process
# ==========================
uploaded_file = st.file_uploader("Upload document (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.getvalue()
        st.session_state.doc_id = sha256_bytes(pdf_bytes)
        st.session_state.doc_filename = uploaded_file.name
        st.session_state.page_texts = extract_pdf_page_texts(pdf_bytes)

        # build full text with page tags (for summary)
        full = []
        for p in st.session_state.page_texts:
            full.append(f"[PAGE {p['page']}]\n{p['text']}")
        st.session_state.doc_text = "\n\n".join(full).strip()
        st.success(f"Document loaded: {len(st.session_state.doc_text)} characters")
    else:
        txt = uploaded_file.read().decode("utf-8", errors="replace")
        st.session_state.doc_id = sha256_bytes(txt.encode("utf-8"))
        st.session_state.doc_filename = uploaded_file.name
        st.session_state.page_texts = [{"page": 1, "text": txt}]
        st.session_state.doc_text = txt
        st.success(f"Document loaded: {len(txt)} characters")

    # mark as not processed until user clicks
    st.session_state.processed = False

if st.button("Process Document"):
    if not st.session_state.doc_text:
        st.warning("Upload a document first.")
    else:
        with st.spinner("Indexing document into KB..."):
            embeddings = get_embeddings()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )

            all_texts = []
            all_metas = []

            doc_id = st.session_state.doc_id
            filename = st.session_state.doc_filename or "unknown"

            for page_item in st.session_state.page_texts:
                page_num = page_item["page"]
                page_txt = page_item["text"] or ""
                if not page_txt.strip():
                    continue
                pieces = splitter.split_text(page_txt)
                for piece in pieces:
                    all_texts.append(piece)
                    all_metas.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page_num,
                        "chunk_type": "main_text",
                        "indexed_at": now_iso(),
                    })

            if not all_texts:
                st.error("No text could be extracted from this document.")
            else:
                # Build/store vectorstore
                if kb_backend.startswith("Qdrant"):
                    qc = get_qdrant_client()
                    if qc is None:
                        st.warning("Qdrant credentials missing. Using FAISS instead.")
                        vs = FAISS.from_texts(all_texts, embeddings, metadatas=all_metas)
                        save_faiss(vs)
                        st.session_state.vectorstore = vs
                    else:
                        collection = get_secret("QDRANT_COLLECTION", "doc_kb")
                        ensure_qdrant_collection(qc, collection)
                        vs = QdrantVectorStore(
                            client=qc,
                            collection_name=collection,
                            embedding=embeddings
                        )
                        vs.add_texts(all_texts, metadatas=all_metas)
                        st.session_state.vectorstore = vs
                else:
                    # FAISS
                    vs = FAISS.from_texts(all_texts, embeddings, metadatas=all_metas)
                    save_faiss(vs)
                    st.session_state.vectorstore = vs

                st.session_state.processed = True
                st.success(f"Indexed {len(all_texts)} chunks. You can now query this document.")


# ==========================
# Tabs: Summary + Q&A
# ==========================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])

with tab1:
    st.markdown("### Summary")
    if st.button("Generate Summary"):
        if not st.session_state.doc_text:
            st.warning("Upload a document first.")
        else:
            with st.spinner("Summarizing..."):
                msgs = [
                    {"role": "system", "content": "Summarize faithfully and avoid adding facts. Provide 8-12 bullet points."},
                    {"role": "user", "content": st.session_state.doc_text[:9000]}
                ]
                out = llm_chat_with_fallback(
                    model_id=model_id, messages=msgs,
                    temperature=temperature, max_tokens=min(1200, max_tokens),
                    hf_token=hf_token, groq_key=groq_key, debug=debug
                )
                st.write(out or "No summary returned.")


with tab2:
    st.markdown("### Q&A (Ask questions about the processed document)")

    # show chat history
    for m in st.session_state.chat_history:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    question = st.chat_input("Ask something about the processed document...")
    if question:
        # must be processed to query
        if not st.session_state.processed or st.session_state.vectorstore is None:
            ans = "⚠️ Please upload and click **Process Document** first, then ask your question."
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            append_message(SESSION_ID, "assistant", ans)
            st.rerun()

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        append_message(SESSION_ID, "user", question)

        # Retrieve a larger set then filter by doc_id (works for Qdrant + FAISS)
        doc_id = st.session_state.doc_id
        raw_k = max(20, int(top_k) * 5)

        docs = st.session_state.vectorstore.similarity_search(question, k=raw_k)

        filtered = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            if md.get("doc_id") == doc_id:
                filtered.append(d)

        # if filtering yields too little, fallback to top_k unfiltered with warning
        if len(filtered) < int(top_k):
            filtered = docs[:int(top_k)]
            warning = "⚠️ Retrieved results may include older docs in the KB. Consider using a fresh collection per doc if needed."
        else:
            filtered = filtered[:int(top_k)]
            warning = ""

        # Build sources sidebar info + context with page refs
        sources = []
        ctx_parts = []
        for d in filtered:
            md = getattr(d, "metadata", {}) or {}
            page = md.get("page", "?")
            filename = md.get("filename", "document")
            chunk = getattr(d, "page_content", "") or ""
            preview = (chunk.strip().replace("\n", " ")[:220] + ("…" if len(chunk) > 220 else ""))
            sources.append({"filename": filename, "page": page, "preview": preview})
            ctx_parts.append(f"(Page {page}) {chunk}")

        st.session_state.last_sources = sources

        context = "\n\n".join(ctx_parts)

        # Ask LLM grounded with page citations
        system = (
            "Answer ONLY using the provided context excerpts.\n"
            "When you use information, cite it like (Page X).\n"
            "If not in context, say: Not found in the processed document."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ]

        with st.spinner("Answering..."):
            answer = llm_chat_with_fallback(
                model_id=model_id, messages=msgs,
                temperature=temperature, max_tokens=max_tokens,
                hf_token=hf_token, groq_key=groq_key, debug=debug
            )
            answer = answer or "No answer returned."

        if warning:
            answer = warning + "\n\n" + answer

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)
        st.rerun()
