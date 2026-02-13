import os
import re
import io
import time
import base64
import sqlite3
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "📄 Document Summarization & Q&A (Unified: DeepSeek OCR2 + LLaVA + Groq routing)"
HF_INFER_BASE = "https://api-inference.huggingface.co/v1/"
DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"
LLAVA_MODEL = "llava-hf/llava-v1.6-34b-hf"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

CHAT_DB_PATH = "chat_history.sqlite"
FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(page_title="Document Summarization & Q&A", layout="wide")
st.title(APP_TITLE)


# =========================================================
# Helpers: env/secrets
# =========================================================
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

def normalize_text(s: str) -> str:
    return (s or "").strip()

def looks_truncated(s: str) -> bool:
    if not s:
        return False
    tail = s.strip()[-40:]
    if tail.endswith("...") or tail.endswith(".."):
        return True
    if re.search(r"[A-Za-z0-9]$", tail) and not re.search(r"[.!?]\s*$", tail):
        return True
    return False


# =========================================================
# UI: Cache-bust reload (mitigates Streamlit hashed JS cache mismatch after deploy)
# =========================================================
with st.sidebar:
    if st.button("🔄 Hard Reload (Cache Bust)"):
        st.query_params["v"] = str(int(time.time()))
        st.rerun()


# =========================================================
# SQLite chat memory
# =========================================================
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
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()

def clear_history(session_id: str):
    con = sqlite3.connect(CHAT_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
    con.commit()
    con.close()


# =========================================================
# Session state
# =========================================================
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("session_id", str(uuid.uuid4()))
SESSION_ID = st.session_state.session_id

ss_init("vectorstore", None)
ss_init("kb_backend", "Qdrant Cloud (API key)")

# The robust "indexed marker" (fixes your persistent issue)
ss_init("indexed_doc_id", None)
ss_init("indexed_chunks_count", 0)

ss_init("text", "")
ss_init("page_texts", [])
ss_init("chunks", [])

ss_init("last_pdf_bytes", None)
ss_init("last_pdf_name", None)

ss_init("chat_history", [])
ss_init("last_sources", [])

ss_init("debug_raw", False)

init_chat_db()
if "history_synced" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)
    st.session_state.history_synced = True


# =========================================================
# Vectorstore helpers
# =========================================================
def save_faiss(vectorstore: FAISS, path: Path = FAISS_DIR):
    vectorstore.save_local(str(path))

def load_faiss(embeddings, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


# =========================================================
# Qdrant init
# =========================================================
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def init_qdrant_vectorstore() -> Tuple[Optional[QdrantVectorStore], Optional[str]]:
    url = get_secret("QDRANT_URL", None)
    api_key = get_secret("QDRANT_API_KEY", None)
    collection_name = get_secret("QDRANT_COLLECTION", "doc_kb")

    if not url or not api_key:
        return None, "Missing QDRANT_URL or QDRANT_API_KEY"

    try:
        client = QdrantClient(url=url, api_key=api_key)
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        vs = QdrantVectorStore(client=client, collection_name=collection_name, embedding=EMBEDDINGS)
        return vs, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ensure_vectorstore(kb_backend: str):
    if kb_backend.startswith("Qdrant"):
        vs, err = init_qdrant_vectorstore()
        if err:
            st.warning(f"Qdrant not available: {err}. Falling back to FAISS.")
            local = load_faiss(EMBEDDINGS)
            st.session_state.vectorstore = local
        else:
            st.session_state.vectorstore = vs
    else:
        local = load_faiss(EMBEDDINGS)
        st.session_state.vectorstore = local


# =========================================================
# Text LLM: HF Router->Groq primary + Groq fallback
# =========================================================
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, resp

def llm_chat_with_fallback(model_id: str, messages, temperature: float, max_tokens: int,
                           hf_token: Optional[str], groq_key: Optional[str], debug: bool = False) -> str:
    # Primary HF Router -> Groq
    if hf_token:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            content, _ = openai_chat(hf_client, routed, messages, temperature, max_tokens)
            if normalize_text(content):
                return normalize_text(content)
        except Exception:
            if debug:
                st.code(traceback.format_exc())

    # Fallback Groq direct
    if groq_key:
        try:
            groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            content, _ = openai_chat(groq_client, model_id, messages, temperature, max_tokens)
            return normalize_text(content)
        except Exception:
            if debug:
                st.code(traceback.format_exc())

    return ""

def llm_chat_with_continue(model_id: str, messages, temperature: float, max_tokens: int,
                           hf_token: Optional[str], groq_key: Optional[str]) -> str:
    out1 = llm_chat_with_fallback(model_id, messages, temperature, max_tokens, hf_token, groq_key)
    if looks_truncated(out1):
        cont = messages + [{"role": "user", "content": "Continue from where you left off. Do not repeat previous text."}]
        out2 = llm_chat_with_fallback(model_id, cont, temperature, max_tokens, hf_token, groq_key)
        if out2 and out2 not in out1:
            return (out1 + "\n" + out2).strip()
    return out1


# =========================================================
# HF Vision calls (DeepSeek + LLaVA)
# =========================================================
def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    img = img.convert("RGB")
    max_side = 1300
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    quality = 78
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    while len(data) > max_bytes and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def hf_vision_chat(hf_token: str, model: str, img: Image.Image, prompt: str,
                   max_tokens: int = 1200, retries: int = 2) -> str:
    if not hf_token:
        return ""
    data_url = image_to_data_url(img)
    client = OpenAI(base_url=HF_INFER_BASE, api_key=hf_token)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    last_err = ""
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            return (out or "").strip()
        except Exception as e:
            last_err = str(e)
            if "413" in last_err or "Payload Too Large" in last_err:
                data_url = image_to_data_url(img, max_bytes=450_000)
                messages[0]["content"][0]["image_url"]["url"] = data_url
            time.sleep(1.0)
    return ""

def hf_vision_chat_with_continue(hf_token: str, model: str, img: Image.Image, prompt: str,
                                 max_tokens: int = 1200) -> str:
    out1 = hf_vision_chat(hf_token, model, img, prompt, max_tokens=max_tokens, retries=2)
    if looks_truncated(out1):
        out2 = hf_vision_chat(hf_token, model, img, prompt + "\n\nContinue from where you left off. Do not repeat.",
                              max_tokens=max_tokens, retries=2)
        if out2 and out2 not in out1:
            return (out1 + "\n" + out2).strip()
    return out1


# =========================================================
# Parse Table/Page
# =========================================================
TABLE_REF_RE = re.compile(r"\b(?:table|tab\.?)\s*(\d+)\b", re.IGNORECASE)
PAGE_REF_RE = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)

def parse_table_number(q: str) -> Optional[int]:
    m = TABLE_REF_RE.search(q or "")
    return int(m.group(1)) if m else None

def parse_page_number(q: str) -> Optional[int]:
    m = PAGE_REF_RE.search(q or "")
    return int(m.group(1)) if m else None

def table_pat(n: int):
    return re.compile(rf"\b(tab(?:le)?\.?)\s*{n}\b", re.IGNORECASE)

def contains_specific_table(text: str, n: int) -> bool:
    return bool(table_pat(n).search(text or ""))


# =========================================================
# PDF Rendering
# =========================================================
def render_pdf_page(pdf_bytes: bytes, page_num_1based: int, dpi: int = 220) -> Optional[Image.Image]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        idx = page_num_1based - 1
        if idx < 0 or idx >= doc.page_count:
            doc.close()
            return None
        page = doc.load_page(idx)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception:
        return None


# =========================================================
# Summary prompt
# =========================================================
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 8-12 bullet points."},
        {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
    ]


# =========================================================
# Runtime: extract + explain table
# =========================================================
def deepseek_extract_table_md(hf_token: str, img: Image.Image, table_n: int) -> str:
    prompt = f"<image>\n<|grounding|>Convert the document to markdown.\nFocus on Table {table_n}."
    return hf_vision_chat_with_continue(hf_token, DEEPSEEK_OCR_MODEL, img, prompt, max_tokens=1800)

def llava_explain_table(hf_token: str, img: Image.Image, table_md: str, table_n: int) -> str:
    prompt = (
        f"This image contains Table {table_n}. Explain the table.\n"
        "Use the extracted table text below as grounding:\n\n"
        f"{table_md}\n\n"
        "Explain columns, key values, patterns, comparisons, and takeaway."
    )
    return hf_vision_chat_with_continue(hf_token, LLAVA_MODEL, img, prompt, max_tokens=1200)


# =========================================================
# Sidebar controls + sources
# =========================================================
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local fallback)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 12, 6, 1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 4096, 1024, 64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
st.session_state.debug_raw = st.sidebar.checkbox("Debug", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.", icon="🧹")

st.sidebar.markdown("---")
st.sidebar.subheader("Sources (last answer)")
if st.session_state.last_sources:
    for i, src in enumerate(st.session_state.last_sources, start=1):
        with st.sidebar.expander(f"{i}. {src.get('filename','document')} — Page {src.get('page','?')}"):
            st.caption(src.get("chunk_type", ""))
            st.write(src.get("preview", ""))
            if st.session_state.last_pdf_bytes and src.get("page"):
                img = render_pdf_page(st.session_state.last_pdf_bytes, int(src["page"]), dpi=160)
                if img is not None:
                    st.image(img, caption=f"Page {src['page']} preview", use_container_width=True)
else:
    st.sidebar.caption("No sources yet.")

if st.session_state.last_pdf_bytes and st.session_state.last_pdf_name:
    st.sidebar.download_button(
        "⬇️ Download current PDF",
        data=st.session_state.last_pdf_bytes,
        file_name=st.session_state.last_pdf_name,
        mime="application/pdf",
        help="Download PDF and jump to page numbers in your viewer."
    )


# =========================================================
# Tokens + initialize store
# =========================================================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)

ensure_vectorstore(kb_backend)


# =========================================================
# Upload + Process document
# =========================================================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        # create a stable doc_id for this upload
        doc_id = f"{uploaded_file.name}:{len(uploaded_file.getvalue())}"
        st.session_state.current_doc_id = doc_id

        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            st.session_state.last_pdf_bytes = pdf_bytes
            st.session_state.last_pdf_name = uploaded_file.name

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_texts = []
            full_text_parts = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                t = page.get_text("text") or ""
                page_texts.append({"page": i + 1, "text": t})
                full_text_parts.append(f"[PAGE {i+1}]\n{t}")
            doc.close()

            st.session_state.page_texts = page_texts
            st.session_state.text = "\n\n".join(full_text_parts).strip()
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")
            st.session_state.page_texts = []
            st.session_state.last_pdf_bytes = None
            st.session_state.last_pdf_name = None

        st.success(f"Document loaded: {len(st.session_state.text)} characters")
        # reset indexed markers for new upload
        st.session_state.indexed_doc_id = None
        st.session_state.indexed_chunks_count = 0
        st.session_state.doc_processed = False

    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB..."):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

            all_chunks = []
            all_metas = []

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown") if uploaded_file else "unknown",
                "filetype": getattr(uploaded_file, "type", "unknown") if uploaded_file else "unknown",
                "uploaded_at": datetime.utcnow().isoformat(),
                "doc_id": st.session_state.get("current_doc_id"),
            }

            if st.session_state.page_texts:
                for item in st.session_state.page_texts:
                    p = int(item["page"])
                    txt = item["text"] or ""
                    if not txt.strip():
                        continue
                    pieces = splitter.split_text(txt)
                    for ch in pieces:
                        all_chunks.append(ch)
                        m = dict(meta_base)
                        m["chunk_type"] = "main_text"
                        m["page"] = p
                        all_metas.append(m)
            else:
                pieces = splitter.split_text(st.session_state.text)
                for ch in pieces:
                    all_chunks.append(ch)
                    m = dict(meta_base)
                    m["chunk_type"] = "main_text"
                    m["page"] = None
                    all_metas.append(m)

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_texts(all_chunks, EMBEDDINGS, metadatas=all_metas)
                save_faiss(st.session_state.vectorstore)
            else:
                st.session_state.vectorstore.add_texts(all_chunks, metadatas=all_metas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)

            # ✅ critical: set robust marker that indexing succeeded
            st.session_state.doc_processed = True
            st.session_state.indexed_doc_id = st.session_state.get("current_doc_id")
            st.session_state.indexed_chunks_count = len(all_chunks)

            st.success(f"KB indexed for this document: {len(all_chunks)} chunks ✅")
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)


# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# =========================================================
# Summary Tab
# =========================================================
with tab1:
    st.markdown("**Generate a summary of the uploaded document.**")
    max_input_chars = st.slider("Max chars for summary input", 256, 50000, 9000, 256)

    if st.button("Generate Summary"):
        if not st.session_state.text:
            st.warning("Upload a document first.")
        else:
            text_in = st.session_state.text[: int(max_input_chars)]
            msgs = build_summary_prompt(text_in)
            summary = llm_chat_with_continue(model_id, msgs, temperature, min(max_tokens, 1200), hf_token, groq_key)
            st.write(summary or "No summary returned.")


# =========================================================
# Q&A Tab (fixed indexing check + table 3 fallback)
# =========================================================
with tab2:
    st.markdown("**Ask about your document. For tables: “Explain Table 3 on page 7”.**")

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    user_question = st.chat_input("Ask a question…")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        append_message(SESSION_ID, "user", user_question)

        # ✅ Robust indexing check: based on our ingest marker, not Qdrant count
        current_doc = st.session_state.get("current_doc_id")
        if not (st.session_state.doc_processed and st.session_state.indexed_doc_id == current_doc and st.session_state.indexed_chunks_count > 0):
            warn = "⚠️ Please upload and click **Process Document** first so the KB is indexed for this document."
            st.session_state.chat_history.append({"role": "assistant", "content": warn})
            append_message(SESSION_ID, "assistant", warn)
            st.rerun()

        ensure_vectorstore(kb_backend)
        if st.session_state.vectorstore is None:
            answer = "Knowledge base not ready (vector store missing)."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.rerun()

        table_n = parse_table_number(user_question)
        page_n = parse_page_number(user_question)

        # Retrieve docs
        docs = []
        try:
            docs = st.session_state.vectorstore.similarity_search(user_question, k=int(top_k))
        except Exception:
            docs = []

        # Build context + sources
        context_lines = []
        sources = []
        for d in docs:
            txt = getattr(d, "page_content", "") or ""
            md = getattr(d, "metadata", {}) or {}
            page = md.get("page", None)
            filename = md.get("filename", st.session_state.last_pdf_name or "document")
            chunk_type = md.get("chunk_type", "main_text")
            preview = (txt.strip().replace("\n", " ")[:280] + ("…" if len(txt) > 280 else ""))
            sources.append({"filename": filename, "page": page, "chunk_type": chunk_type, "preview": preview})

            page_tag = f"Page {page}" if page else "Page ?"
            context_lines.append(f"[{page_tag} | {filename}]\n{txt}")

        st.session_state.last_sources = sources[:8]

        # Runtime table fallback if table requested but not in retrieved context
        if table_n is not None and st.session_state.last_pdf_bytes and hf_token:
            combined = "\n".join(context_lines)
            if not contains_specific_table(combined, table_n):
                # candidate pages: page hint first + neighbors; then pages containing "Table n" in text; else first 30 pages
                candidate_pages: List[int] = []
                if page_n is not None:
                    candidate_pages += [int(page_n), max(1, int(page_n)-1), int(page_n)+1]

                for item in st.session_state.page_texts:
                    p = int(item["page"])
                    if contains_specific_table(item.get("text", ""), table_n) and p not in candidate_pages:
                        candidate_pages.append(p)
                        if p+1 not in candidate_pages:
                            candidate_pages.append(p+1)

                if not candidate_pages:
                    try:
                        doc = fitz.open(stream=st.session_state.last_pdf_bytes, filetype="pdf")
                        limit = min(30, doc.page_count)
                        doc.close()
                    except Exception:
                        limit = 30
                    candidate_pages = list(range(1, limit+1))

                best_page = None
                best_md = ""
                for p in candidate_pages[:20]:
                    img = render_pdf_page(st.session_state.last_pdf_bytes, p, dpi=300)
                    if img is None:
                        continue
                    md = deepseek_extract_table_md(hf_token, img, table_n)
                    if md and ("|" in md or f"Table {table_n}" in md or f"TABLE {table_n}" in md):
                        best_page = p
                        best_md = md
                        break

                if best_page is not None:
                    img = render_pdf_page(st.session_state.last_pdf_bytes, best_page, dpi=300)
                    expl = llava_explain_table(hf_token, img, best_md, table_n)
                    runtime_answer = (
                        f"### ✅ Table {table_n} (Page {best_page})\n\n"
                        f"#### Extracted (DeepSeek OCR‑2)\n{best_md}\n\n"
                        f"#### Explanation (LLaVA)\n{expl}\n\n"
                        f"**Source:** {st.session_state.last_pdf_name or 'document'} (Page {best_page})"
                    )

                    # Store runtime chunk
                    try:
                        runtime_chunk = f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {best_page}] {runtime_answer}"
                        meta = {
                            "source": "runtime_ocr",
                            "filename": st.session_state.last_pdf_name or "document",
                            "filetype": "application/pdf",
                            "uploaded_at": datetime.utcnow().isoformat(),
                            "chunk_type": "table_runtime",
                            "page": best_page,
                            "table_number": table_n,
                            "doc_id": st.session_state.get("current_doc_id"),
                        }
                        st.session_state.vectorstore.add_texts([runtime_chunk], metadatas=[meta])
                        if isinstance(st.session_state.vectorstore, FAISS):
                            save_faiss(st.session_state.vectorstore)
                    except Exception:
                        pass

                    st.session_state.last_sources = [{
                        "filename": st.session_state.last_pdf_name or "document",
                        "page": best_page,
                        "chunk_type": "table_runtime",
                        "preview": "Runtime extracted Table content + explanation."
                    }]

                    st.session_state.chat_history.append({"role": "assistant", "content": runtime_answer})
                    append_message(SESSION_ID, "assistant", runtime_answer)
                    st.rerun()

        # Normal grounded answer with page citations (no S1/S2)
        if not context_lines:
            answer = "Not found in the document excerpts. Increase Top K or ask with a page number."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.rerun()

        system = (
            "You are a precise assistant. Use ONLY the provided excerpts.\n"
            "Cite sources as (Page X) based on excerpt headers.\n"
            "Do NOT cite S1/S2. Do NOT invent page numbers.\n"
            "If not present, say: 'Not found in the document excerpts.'"
        )
        user = f"Document excerpts:\n\n{'\n\n'.join(context_lines)}\n\nQuestion: {user_question}\nAnswer:"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        answer = llm_chat_with_continue(model_id, msgs, temperature, max_tokens, hf_token, groq_key)
        answer = answer or "No answer returned."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)
        st.rerun()
