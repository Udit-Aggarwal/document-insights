# app.py
"""
Complete unified app:
- Upload PDF/TXT
- Process document -> chunk + embed + store KB (Qdrant or FAISS fallback)
- Summary tab (restored)
- Q&A tab with SQLite chat memory
- Text LLM routing: HF Router -> Groq primary, Groq direct fallback

Vision enhancements (HF token):
- deepseek-ai/DeepSeek-OCR-2: OCR/table extraction/markdown conversion prompts like "<|grounding|>Convert the document to markdown." [1](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-embeddings/)[2](https://stackoverflow.com/questions/74721623/how-do-you-use-pymongo-to-connect-to-mongodb-atlas)
- llava-hf/llava-v1.6-34b-hf: explanation / visual reasoning; model expects image + prompt. [3](https://github.com/groq/groq-python)

Robust fixes:
1) Table 3 vs Table 9 bug:
   - runtime fallback triggers only if the requested table number is missing (not just "some table exists").
   - locate table pages using page-wise PDF text search first; then OCR those pages.
2) References:
   - stop S1/S2 in answers
   - ask model to cite (Page X)
   - show sources used in sidebar with page preview and PDF download
3) Completeness:
   - continuation pass for vision outputs if truncated
   - larger token slider for text LLM
"""

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
from qdrant_client.http import models as qmodels


# =========================
# Page config
# =========================
st.set_page_config(page_title="Document Summarization & Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A (DeepSeek OCR2 + LLaVA + Groq routing)")


# =========================
# Local FAISS persistence (fallback KB)
# =========================
FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)

def save_faiss(vectorstore: FAISS, path: Path = FAISS_DIR):
    vectorstore.save_local(str(path))

def load_faiss(embeddings, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


# =========================
# SQLite chat history persistence
# =========================
CHAT_DB_PATH = "chat_history.sqlite"

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


# =========================
# Session state
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "page_texts" not in st.session_state:
    st.session_state.page_texts = []  # list of {"page": int, "text": str}
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False

# chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []  # list of dicts: {page, filename, chunk_type, preview}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# store pdf bytes for runtime fallback + page preview
if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None
if "last_pdf_name" not in st.session_state:
    st.session_state.last_pdf_name = None

SESSION_ID = st.session_state.session_id
init_chat_db()
if "history_synced" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)
    st.session_state.history_synced = True


# =========================
# Helpers
# =========================
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

def normalize_text(s: str) -> str:
    return (s or "").strip()


# =========================
# Text LLM routing (HF Router -> Groq primary, Groq direct fallback)
# =========================
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, resp

def llm_chat_with_fallback(
    model_id: str, messages, temperature: float, max_tokens: int,
    hf_token: Optional[str], groq_key: Optional[str], debug: bool = False
):
    result = {"content": "", "primary_used": False, "raw": None, "error_primary": None, "error_fallback": None}

    # Primary: HF Router -> Groq provider
    if hf_token:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            content, raw = openai_chat(hf_client, routed, messages, temperature, max_tokens)
            if normalize_text(content):
                result.update({"content": content, "primary_used": True, "raw": raw})
                return result
            result["error_primary"] = "Primary returned empty content."
        except Exception as e:
            result["error_primary"] = f"{type(e).__name__}: {e}"
            if debug:
                st.code(traceback.format_exc())
    else:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"

    # Fallback: Groq direct
    if not groq_key:
        result["error_fallback"] = "Missing GROQ_API_KEY (fallback not available)"
        return result

    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        content, raw = openai_chat(groq_client, model_id, messages, temperature, max_tokens)
        result.update({"content": content, "raw": raw})
        return result
    except Exception as e:
        result["error_fallback"] = f"{type(e).__name__}: {e}"
        if debug:
            st.code(traceback.format_exc())
        return result


# =========================
# HF vision calls (DeepSeek OCR2 + LLaVA) with compression + continuation
# =========================
HF_INFER_BASE = "https://api-inference.huggingface.co/v1/"
DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"  # [1](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-embeddings/)[2](https://stackoverflow.com/questions/74721623/how-do-you-use-pymongo-to-connect-to-mongodb-atlas)
LLAVA_MODEL = "llava-hf/llava-v1.6-34b-hf"        # [3](https://github.com/groq/groq-python)

def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    """Compress + downscale to reduce 413 Payload Too Large risk. [4](https://www.apideck.com/blog/how-to-get-your-groq-api-key)"""
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
    """HF Inference OpenAI-compatible call with image_url + text blocks."""
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

def looks_truncated(s: str) -> bool:
    if not s:
        return False
    tail = s.strip()[-40:]
    if tail.endswith("...") or tail.endswith(".."):
        return True
    if re.search(r"[A-Za-z0-9]$", tail) and not re.search(r"[.!?]\s*$", tail):
        return True
    return False

def hf_vision_chat_with_continue(hf_token: str, model: str, img: Image.Image, prompt: str,
                                 max_tokens: int = 1200) -> str:
    out1 = hf_vision_chat(hf_token, model, img, prompt, max_tokens=max_tokens, retries=2)
    if looks_truncated(out1):
        out2 = hf_vision_chat(hf_token, model, img,
                              prompt + "\n\nContinue from where you left off. Do not repeat.",
                              max_tokens=max_tokens, retries=2)
        if out2 and out2 not in out1:
            return (out1 + "\n" + out2).strip()
    return out1


# =========================
# Parse references (table/figure/page) + BUGFIX: ensure runtime triggers only if requested number missing
# =========================
TABLE_REF_RE = re.compile(r"\b(?:table|tab\.?)\s*(\d+)\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(?:figure|fig\.?)\s*(\d+)\b", re.IGNORECASE)
PAGE_REF_RE  = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)

def parse_table_number(q: str) -> Optional[int]:
    m = TABLE_REF_RE.search(q or "")
    return int(m.group(1)) if m else None

def parse_figure_number(q: str) -> Optional[int]:
    m = FIGURE_REF_RE.search(q or "")
    return int(m.group(1)) if m else None

def parse_page_number(q: str) -> Optional[int]:
    m = PAGE_REF_RE.search(q or "")
    return int(m.group(1)) if m else None

def is_visual_question(q: str) -> bool:
    return bool(re.search(r"\b(table|tab|figure|fig|diagram|chart|graph|image)\b", q or "", flags=re.I))

def table_pat(n: int):
    return re.compile(rf"\b(tab(?:le)?\.?)\s*{n}\b", re.IGNORECASE)

def figure_pat(n: int):
    return re.compile(rf"\b(fig(?:ure)?\.?)\s*{n}\b", re.IGNORECASE)

def has_specific_table_in_text(text: str, table_n: int) -> bool:
    return bool(table_pat(table_n).search(text or ""))

def has_specific_figure_in_text(text: str, fig_n: int) -> bool:
    return bool(figure_pat(fig_n).search(text or ""))


# =========================
# PDF page rendering (for runtime extraction + sidebar preview)
# =========================
def render_pdf_page(pdf_bytes: bytes, page_num_1based: int, dpi: int = 200) -> Optional[Image.Image]:
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


# =========================
# KB init: Qdrant + FAISS fallback
# =========================
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

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
        if not isinstance(st.session_state.vectorstore, QdrantVectorStore):
            vs, err = init_qdrant_vectorstore()
            if err:
                st.warning(f"Qdrant not available: {err}. Falling back to FAISS.")
                local = load_faiss(EMBEDDINGS)
                if local is not None:
                    st.session_state.vectorstore = local
                    st.session_state.kb_ready = True
                else:
                    st.session_state.vectorstore = None
                    st.session_state.kb_ready = False
            else:
                st.session_state.vectorstore = vs
                st.session_state.kb_ready = True
        else:
            st.session_state.kb_ready = True
    else:
        if not isinstance(st.session_state.vectorstore, FAISS):
            local = load_faiss(EMBEDDINGS)
            if local is not None:
                st.session_state.vectorstore = local
                st.session_state.kb_ready = True
            else:
                st.session_state.vectorstore = None
                st.session_state.kb_ready = False
        else:
            st.session_state.kb_ready = True


# =========================
# Retrieval helpers + source collection (no S1/S2)
# =========================
VISUAL_TYPES = {"table_runtime", "figure_runtime", "page_ocr", "main_text"}

def qdrant_filter_for_chunk_types(types: List[str]) -> Optional[qmodels.Filter]:
    should = []
    for path in ["metadata.chunk_type", "chunk_type"]:
        should.append(qmodels.FieldCondition(key=path, match=qmodels.MatchAny(any=types)))
    return qmodels.Filter(should=should)

def retrieve_docs(vectorstore, query: str, k: int) -> List[Any]:
    # simplest: use similarity_search
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        return []


# =========================
# Summary helper (restored)
# =========================
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 8-12 bullet points."},
        {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
    ]


# =========================
# Runtime extractors (dual model)
# =========================
def runtime_extract_table_md(hf_token: str, img: Image.Image, table_n: int) -> str:
    prompt = f"<image>\n<|grounding|>Convert the document to markdown.\nFocus on Table {table_n}."
    return hf_vision_chat_with_continue(hf_token, DEEPSEEK_OCR_MODEL, img, prompt, max_tokens=1800)

def runtime_explain_table(hf_token: str, img: Image.Image, table_md: str, table_n: int) -> str:
    prompt = (
        f"This image contains Table {table_n}. Explain the table in detail.\n"
        "Use the extracted markdown table below as grounding:\n\n"
        f"{table_md}\n\n"
        "Explain: columns meaning, key values, comparisons, patterns, and takeaway."
    )
    return hf_vision_chat_with_continue(hf_token, LLAVA_MODEL, img, prompt, max_tokens=1200)


# =========================
# Sidebar controls + Sources panel
# =========================
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
st.sidebar.header("Chat Memory")
max_history_turns = st.sidebar.slider("Max chat turns kept", 2, 20, 8, 1)
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []
    st.session_state.last_context_blocks = []
    st.session_state.last_ref = None
    st.session_state.last_sources = []
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.", icon="🧹")

st.sidebar.markdown("---")
st.sidebar.subheader("Sources (last answer)")
if st.session_state.last_sources:
    for i, src in enumerate(st.session_state.last_sources, start=1):
        with st.sidebar.expander(f"{i}. {src.get('filename','document')} — Page {src.get('page','?')}"):
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
        help="Download and open locally; you can jump to a page number in your PDF viewer."
    )


# =========================
# Tokens
# =========================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. Runtime table extraction won't work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. Groq fallback won't work if HF Router fails.")


# =========================
# Ensure KB ready
# =========================
ensure_vectorstore(kb_backend)


# =========================
# Upload + Process Document (page-aware chunking)
# =========================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
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
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB (page-aware)..."):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )

            all_chunks = []
            all_metas = []

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown") if uploaded_file else "unknown",
                "filetype": getattr(uploaded_file, "type", "unknown") if uploaded_file else "unknown",
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            if st.session_state.page_texts:
                # PAGE-AWARE chunking: preserve page numbers for references
                for item in st.session_state.page_texts:
                    p = int(item["page"])
                    txt = item["text"] or ""
                    if not txt.strip():
                        continue
                    chunks = splitter.split_text(txt)
                    for ch in chunks:
                        all_chunks.append(ch)
                        m = dict(meta_base)
                        m["chunk_type"] = "main_text"
                        m["page"] = p
                        all_metas.append(m)
            else:
                chunks = splitter.split_text(st.session_state.text)
                for ch in chunks:
                    all_chunks.append(ch)
                    m = dict(meta_base)
                    m["chunk_type"] = "main_text"
                    m["page"] = None
                    all_metas.append(m)

            st.session_state.chunks = all_chunks

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_texts(all_chunks, EMBEDDINGS, metadatas=all_metas)
                save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_chunks)} chunks in FAISS KB.")
            else:
                st.session_state.vectorstore.add_texts(all_chunks, metadatas=all_metas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_chunks)} chunks in KB.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)


# =========================
# Tabs: Summary + Q&A
# =========================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# =========================
# Summary Tab (restored)
# =========================
with tab1:
    st.markdown("**Generate a summary of the uploaded document.**")
    use_chunked_summary = st.checkbox("Use chunked summary (better for long docs)", value=True)
    max_input_chars = st.slider("Max chars for non-chunked summary", 256, 50000, 8000, 256)

    if st.button("Generate Summary"):
        if not st.session_state.text:
            st.warning("Upload a document first.")
        else:
            with st.spinner("Summarizing..."):
                if use_chunked_summary:
                    if not st.session_state.chunks:
                        st.warning("Click 'Process Document' first for best results (or enable processing).")
                    n = min(12, len(st.session_state.chunks)) if st.session_state.chunks else 0
                    if n == 0:
                        text_in = st.session_state.text[: int(max_input_chars)]
                        result = llm_chat_with_fallback(
                            model_id=model_id,
                            messages=build_summary_prompt(text_in),
                            temperature=temperature,
                            max_tokens=min(max_tokens, 1024),
                            hf_token=hf_token,
                            groq_key=groq_key,
                            debug=st.session_state.debug_raw
                        )
                        st.write(result["content"] or "No summary returned.")
                    else:
                        mini = []
                        for i in range(n):
                            msgs = [
                                {"role": "system", "content": "Summarize in 2-3 bullet points. Avoid adding facts."},
                                {"role": "user", "content": st.session_state.chunks[i]},
                            ]
                            out = llm_chat_with_fallback(
                                model_id=model_id, messages=msgs,
                                temperature=temperature, max_tokens=256,
                                hf_token=hf_token, groq_key=groq_key,
                                debug=st.session_state.debug_raw
                            )
                            mini.append(out["content"] or "")

                        final_msgs = [
                            {"role": "system", "content": "Combine into 8-12 bullet executive summary. No extra facts."},
                            {"role": "user", "content": "\n\n".join(mini)},
                        ]
                        result = llm_chat_with_fallback(
                            model_id=model_id, messages=final_msgs,
                            temperature=temperature, max_tokens=min(max_tokens, 1200),
                            hf_token=hf_token, groq_key=groq_key,
                            debug=st.session_state.debug_raw
                        )
                        st.write(result["content"] or "No summary returned.")
                else:
                    text_in = st.session_state.text[: int(max_input_chars)]
                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=build_summary_prompt(text_in),
                        temperature=temperature,
                        max_tokens=min(max_tokens, 1200),
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw
                    )
                    st.write(result["content"] or "No summary returned.")


# =========================
# Q&A Tab (fix Table 3 + references + UI sources)
# =========================
with tab2:
    st.markdown("**Ask about your document. Tip: “Explain Table 3 on page 7” is deterministic.**")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    user_question = st.chat_input("Ask a question…")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        append_message(SESSION_ID, "user", user_question)

        ensure_vectorstore(kb_backend)
        if st.session_state.vectorstore is None:
            answer = "Knowledge base not ready. Configure Qdrant/FAISS."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.rerun()

        table_n = parse_table_number(user_question)
        fig_n = parse_figure_number(user_question)
        page_n = parse_page_number(user_question)
        prefer_visual = is_visual_question(user_question)

        # Retrieve
        k = max(int(top_k), 10) if prefer_visual else int(top_k)
        docs = retrieve_docs(st.session_state.vectorstore, user_question, k=k)

        # Build sources (metadata-based) and context (no S1/S2 in answer)
        sources = []
        context_lines = []
        for d in docs:
            txt = getattr(d, "page_content", "") or ""
            md = getattr(d, "metadata", {}) or {}
            page = md.get("page", None)
            filename = md.get("filename", st.session_state.last_pdf_name or "document")
            chunk_type = md.get("chunk_type", "main_text")
            preview = (txt.strip().replace("\n", " ")[:280] + ("…" if len(txt) > 280 else ""))

            sources.append({
                "filename": filename,
                "page": page,
                "chunk_type": chunk_type,
                "preview": preview
            })

            # Embed page reference into the context for the model
            # so it can cite (Page X) instead of S1/S2.
            page_tag = f"Page {page}" if page else "Page ?"
            context_lines.append(f"[{page_tag} | {filename} | {chunk_type}]\n{txt}")

        # Reset sidebar sources for this answer
        st.session_state.last_sources = sources[:8]

        # ---- Robust runtime fallback for Table N (fix persistent Table 3) ----
        runtime_answer = None
        if table_n is not None and st.session_state.last_pdf_bytes and hf_token:
            # if Table N not present in retrieved context, run runtime extraction
            combined_context_text = "\n".join(context_lines)
            table_present = has_specific_table_in_text(combined_context_text, table_n)

            if not table_present:
                # 1) find candidate pages from page-wise text first (most reliable)
                candidate_pages = []
                for item in st.session_state.page_texts:
                    if has_specific_table_in_text(item.get("text", ""), table_n):
                        candidate_pages.append(int(item["page"]))

                # if user specified page, prioritize it
                if page_n is not None:
                    candidate_pages = [int(page_n)] + [p for p in candidate_pages if p != int(page_n)]

                # fallback: if no candidates, scan first 25 pages
                if not candidate_pages:
                    try:
                        doc = fitz.open(stream=st.session_state.last_pdf_bytes, filetype="pdf")
                        limit = min(25, doc.page_count)
                        doc.close()
                    except Exception:
                        limit = 25
                    candidate_pages = list(range(1, limit + 1))

                best_page = None
                best_md = ""
                for p in candidate_pages[:12]:  # cap attempts
                    img = render_pdf_page(st.session_state.last_pdf_bytes, p, dpi=300)
                    if img is None:
                        continue
                    md = runtime_extract_table_md(hf_token, img, table_n)
                    # accept if it looks like a table markdown or mentions Table n
                    if md and ("|" in md or has_specific_table_in_text(md, table_n)):
                        best_page = p
                        best_md = md
                        break

                if best_page is not None:
                    img = render_pdf_page(st.session_state.last_pdf_bytes, best_page, dpi=300)
                    expl = runtime_explain_table(hf_token, img, best_md, table_n)

                    runtime_answer = (
                        f"### ✅ Table {table_n} (Page {best_page})\n\n"
                        f"#### Extracted (DeepSeek OCR‑2)\n{best_md}\n\n"
                        f"#### Explanation (LLaVA)\n{expl}\n\n"
                        f"**Source:** {st.session_state.last_pdf_name or 'document'} (Page {best_page})"
                    )

                    # store runtime into KB so future retrieval works
                    try:
                        runtime_chunk = f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {best_page}] {runtime_answer}"
                        meta = {
                            "source": "runtime_ocr",
                            "filename": st.session_state.last_pdf_name or "document",
                            "filetype": "application/pdf",
                            "uploaded_at": datetime.utcnow().isoformat(),
                            "chunk_type": "table_runtime",
                            "page": best_page,
                            "table_number": table_n
                        }
                        st.session_state.vectorstore.add_texts([runtime_chunk], metadatas=[meta])
                        if isinstance(st.session_state.vectorstore, FAISS):
                            save_faiss(st.session_state.vectorstore)
                    except Exception:
                        pass

        # If runtime answer exists, return it directly (avoids S1/S2 + avoids partial answers)
        if runtime_answer:
            st.session_state.chat_history.append({"role": "assistant", "content": runtime_answer})
            append_message(SESSION_ID, "assistant", runtime_answer)

            # update sidebar sources to the correct page
            st.session_state.last_sources = [{
                "filename": st.session_state.last_pdf_name or "document",
                "page": int(re.search(r"Page (\d+)", runtime_answer).group(1)) if re.search(r"Page (\d+)", runtime_answer) else None,
                "chunk_type": "table_runtime",
                "preview": "Runtime extracted Table content + explanation."
            }]
            st.rerun()

        # Otherwise: normal grounded answer, but request page citations not S1/S2
        system = (
            "You are a precise assistant. Use ONLY the provided document excerpts.\n"
            "When you use information, cite it as (Page X) using the page mentioned in the excerpt header.\n"
            "Do NOT cite S1/S2. Do NOT invent page numbers.\n"
            "If the answer is not present, say: 'Not found in the document excerpts.'"
        )
        user = f"Document excerpts:\n\n{'\n\n'.join(context_lines)}\n\nQuestion: {user_question}\nAnswer:"

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = llm_chat_with_fallback(
            model_id=model_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            hf_token=hf_token,
            groq_key=groq_key,
            debug=st.session_state.debug_raw
        )
        answer = normalize_text(result["content"]) or "No answer returned."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)
        st.rerun()
