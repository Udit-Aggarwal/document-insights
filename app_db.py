# app.py
"""
COMPLETE UNIFIED APP (DOES NOT REMOVE ORIGINAL FEATURES)

Original requirements preserved:
✅ Upload PDF/TXT
✅ Process Document -> chunk + embed + store in KB (Qdrant or FAISS fallback)
✅ Summary feature/tab
✅ Q&A (Chat) tab with multi-turn memory (SQLite persistence)
✅ Text LLM routing: HF Router -> Groq (primary) + Groq Direct fallback

Enhancements added (without removing anything):
✅ Dual vision models (HF token):
   - deepseek-ai/DeepSeek-OCR-2: OCR + markdown conversion / structured extraction
   - llava-hf/llava-v1.6-34b-hf: explanation / visual reasoning
✅ Runtime fallback for requested Table N / Figure N:
   - If KB doesn’t contain *that specific* Table N (not just any table), render PDF page and extract+explain.
   - Auto-insert runtime result into KB for future queries.
✅ Completeness:
   - One continuation pass if vision output appears truncated.
✅ Fixes Table3-vs-Table9 issue:
   - Checks for *requested number* before skipping runtime fallback.
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
from typing import Optional, Tuple, List, Dict

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
# Page config (keep original feel)
# =========================
st.set_page_config(page_title="Document Summarization & Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A (Unified: DeepSeek OCR2 + LLaVA + Groq routing)")


# =========================
# Local FAISS persistence (fallback)
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
# SQLite chat memory persistence
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
# Session state defaults
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_context_blocks" not in st.session_state:
    st.session_state.last_context_blocks = []
if "last_ref" not in st.session_state:
    st.session_state.last_ref = None  # "table 3" / "figure 2"

# keep uploaded PDF in-session for runtime fallback
if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None
if "last_pdf_name" not in st.session_state:
    st.session_state.last_pdf_name = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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
# Text LLM routing: HF Router -> Groq (primary) + Groq fallback
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
# Vision models (HF Inference OpenAI-compatible)
# =========================
HF_INFER_BASE = "https://api-inference.huggingface.co/v1/"
DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"
LLAVA_MODEL = "llava-hf/llava-v1.6-34b-hf"

def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    """Compress + downscale to reduce 413 Payload Too Large risk."""
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
    """Call HF OpenAI-compatible endpoint with image_url + text blocks."""
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
    """One continuation pass if output seems truncated."""
    out1 = hf_vision_chat(hf_token, model, img, prompt, max_tokens=max_tokens, retries=2)
    if looks_truncated(out1):
        cont_prompt = (
            prompt
            + "\n\nContinue EXACTLY from where you left off. Do NOT repeat earlier text."
        )
        out2 = hf_vision_chat(hf_token, model, img, cont_prompt, max_tokens=max_tokens, retries=2)
        if out2 and out2 not in out1:
            return (out1 + "\n" + out2).strip()
    return out1


# =========================
# Parse references (Table/Figure/Page)
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

def extract_ref(q: str) -> Optional[str]:
    t = parse_table_number(q)
    if t is not None:
        return f"table {t}"
    f = parse_figure_number(q)
    if f is not None:
        return f"figure {f}"
    return None

def is_visual_question(q: str) -> bool:
    return bool(re.search(r"\b(table|tab|figure|fig|diagram|chart|graph|image)\b", q or "", flags=re.I))


# =========================
# FIX: check for SPECIFIC requested table/figure (solves Table3 vs Table9)
# =========================
def table_pat(n: int):
    return re.compile(rf"\b(tab(?:le)?\.?)\s*{n}\b", re.IGNORECASE)

def figure_pat(n: int):
    return re.compile(rf"\b(fig(?:ure)?\.?)\s*{n}\b", re.IGNORECASE)

def has_specific_table(blocks: List[str], table_n: int) -> bool:
    pat = table_pat(table_n)
    for b in blocks:
        if pat.search(b):
            return True
        if f"[TABLE_RUNTIME][TABLE {table_n}]" in b:
            return True
    return False

def has_specific_table_on_page(blocks: List[str], table_n: int, page_n: Optional[int]) -> bool:
    if page_n is None:
        return has_specific_table(blocks, table_n)
    for b in blocks:
        if f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {page_n}]" in b:
            return True
    # loose match: "Table n" and "page x" both mentioned
    pat = table_pat(table_n)
    for b in blocks:
        if pat.search(b) and re.search(rf"\bpage\s*{page_n}\b", b, re.IGNORECASE):
            return True
    return False

def has_specific_figure(blocks: List[str], fig_n: int) -> bool:
    pat = figure_pat(fig_n)
    for b in blocks:
        if pat.search(b):
            return True
        if f"[FIGURE_RUNTIME][FIGURE {fig_n}]" in b:
            return True
    return False


# =========================
# Render PDF page -> image for runtime fallback
# =========================
def render_pdf_page(pdf_bytes: bytes, page_num_1based: int, dpi: int = 300) -> Optional[Image.Image]:
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
# Retrieval helpers (visual-prioritized)
# =========================
VISUAL_TYPES = {
    "table_runtime", "table_ocr", "table_explain", "table_caption",
    "figure_runtime", "figure_ocr", "figure_explain", "figure_caption",
    "page_ocr"
}

def qdrant_filter_for_chunk_types(types: List[str]) -> Optional[qmodels.Filter]:
    should = []
    for path in ["metadata.chunk_type", "chunk_type"]:
        should.append(qmodels.FieldCondition(key=path, match=qmodels.MatchAny(any=types)))
    return qmodels.Filter(should=should)

def retrieve_docs(vectorstore, query: str, k: int, prefer_visual: bool) -> List:
    docs_all: List = []

    if prefer_visual and isinstance(vectorstore, QdrantVectorStore):
        try:
            flt = qdrant_filter_for_chunk_types(list(VISUAL_TYPES))
            docs_vis = vectorstore.similarity_search(query, k=min(14, max(k, 10)), filter=flt)
            docs_all.extend(docs_vis)
        except Exception:
            pass

    try:
        docs_gen = vectorstore.similarity_search(query, k=max(k, 6))
        docs_all.extend(docs_gen)
    except Exception:
        return []

    # de-dup
    seen = set()
    uniq = []
    for d in docs_all:
        txt = getattr(d, "page_content", "") or ""
        key = txt.strip()[:2000]
        if key and key not in seen:
            seen.add(key)
            uniq.append(d)

    if prefer_visual and not isinstance(vectorstore, QdrantVectorStore):
        vis_first, rest = [], []
        for d in uniq:
            md = getattr(d, "metadata", {}) or {}
            ctype = (md.get("chunk_type") or "").lower()
            if ctype in VISUAL_TYPES:
                vis_first.append(d)
            else:
                rest.append(d)
        uniq = vis_first + rest

    return uniq[:max(k, 10) if prefer_visual else k]


# =========================
# Summary prompt (restored)
# =========================
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 8-12 bullet points."},
        {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
    ]


# =========================
# Runtime extractors (Dual model)
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

def runtime_extract_figure(hf_token: str, img: Image.Image, fig_n: int) -> str:
    prompt = f"<image>\nParse the figure. Figure {fig_n}. Extract labels/axes/legend and key elements."
    return hf_vision_chat_with_continue(hf_token, DEEPSEEK_OCR_MODEL, img, prompt, max_tokens=1400)

def runtime_explain_figure(hf_token: str, img: Image.Image, fig_text: str, fig_n: int) -> str:
    prompt = (
        f"This image contains Figure {fig_n}. Explain it clearly.\n"
        "Use the extracted details below as grounding:\n\n"
        f"{fig_text}\n\n"
        "Now provide: what it shows, axes/legend, key trends, and takeaway."
    )
    return hf_vision_chat_with_continue(hf_token, LLAVA_MODEL, img, prompt, max_tokens=1000)


# =========================
# Sidebar (keep original controls)
# =========================
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local fallback)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 12, 6, 1)
scan_pages_limit = st.sidebar.number_input("Max pages to scan if page not specified", 1, 80, 25, 1)

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
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.", icon="🧹")


# =========================
# Tokens
# =========================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. Runtime table/figure extraction won't work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. Groq fallback won't work if HF Router fails.")


# =========================
# Ensure KB ready
# =========================
ensure_vectorstore(kb_backend)


# =========================
# Upload + Process Document (kept)
# =========================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            st.session_state.last_pdf_bytes = pdf_bytes
            st.session_state.last_pdf_name = uploaded_file.name

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                pages_text.append(page.get_text("text"))
            doc.close()
            st.session_state.text = "\n\n".join(pages_text).strip()

        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")
            st.session_state.last_pdf_bytes = None
            st.session_state.last_pdf_name = None

        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB..."):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap)
            )
            chunks = splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown") if uploaded_file else "unknown",
                "filetype": getattr(uploaded_file, "type", "unknown") if uploaded_file else "unknown",
                "uploaded_at": datetime.utcnow().isoformat(),
                "chunk_type": "main_text",
            }
            metadatas = [dict(meta_base) for _ in chunks]

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_texts(chunks, EMBEDDINGS, metadatas=metadatas)
                save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(chunks)} chunks in FAISS KB.")
            else:
                st.session_state.vectorstore.add_texts(chunks, metadatas=metadatas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(chunks)} chunks in KB.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)


# =========================
# Tabs (Summary kept + Q&A)
# =========================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# =========================
# Summary Tab (kept)
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
                        splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
                        st.session_state.chunks = splitter.split_text(st.session_state.text)

                    n = min(12, len(st.session_state.chunks))
                    mini = []
                    for i in range(n):
                        msgs = [
                            {"role": "system", "content": "Summarize in 2-3 bullet points. Avoid adding facts."},
                            {"role": "user", "content": st.session_state.chunks[i]},
                        ]
                        out = llm_chat_with_fallback(
                            model_id=model_id, messages=msgs,
                            temperature=temperature, max_tokens=min(256, max_tokens),
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
                        temperature=temperature, max_tokens=max_tokens,
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
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw
                    )
                    st.write(result["content"] or "No summary returned.")


# =========================
# Q&A Tab (fixed: table-specific detection + complete runtime output)
# =========================
with tab2:
    st.markdown("**Ask about your document. For deterministic answers use: “Explain Table 3 on page 7”.**")

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    if st.session_state.last_retrieved_docs:
        with st.expander("Retrieved snippets (last turn)"):
            for i, txt in enumerate(st.session_state.last_retrieved_docs, start=1):
                st.markdown(f"**Snippet {i}**")
                st.write(txt[:1500])

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
        ref = extract_ref(user_question)
        prefer_visual = is_visual_question(user_question)

        reuse_prior = (ref is not None and ref == st.session_state.last_ref and len(st.session_state.last_context_blocks) > 0)

        # 1) Retrieve normally
        k = max(int(top_k), 10) if prefer_visual else int(top_k)
        docs = retrieve_docs(st.session_state.vectorstore, user_question, k=k, prefer_visual=prefer_visual)

        context_blocks: List[str] = []
        last_snips: List[str] = []

        for i, d in enumerate(docs, start=1):
            snippet = getattr(d, "page_content", "").strip()
            if snippet:
                context_blocks.append(f"[S{i}] {snippet}")
                last_snips.append(snippet)

        if reuse_prior:
            merged = []
            seen = set()
            for b in (st.session_state.last_context_blocks + context_blocks):
                key = b.strip()[:500]
                if key and key not in seen:
                    seen.add(key)
                    merged.append(b)
            context_blocks = merged

        pdf_bytes = st.session_state.last_pdf_bytes
        pdf_name = st.session_state.last_pdf_name or "uploaded.pdf"

        runtime_text_answer = None
        runtime_chunk_to_store = None
        runtime_meta = None

        # 2) FIXED runtime fallback trigger:
        # Trigger ONLY if the requested table/figure number is missing (not blocked by other tables)
        if table_n is not None and pdf_bytes and hf_token:
            table_missing = not has_specific_table_on_page(context_blocks, table_n, page_n)
            if table_missing:
                # locate page
                target_page = page_n
                if target_page is None:
                    # scan pages using cheap locator on LLaVA
                    try:
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        scan_pages = min(int(scan_pages_limit), doc.page_count)
                        doc.close()
                    except Exception:
                        scan_pages = int(scan_pages_limit)

                    found = None
                    for p in range(1, scan_pages + 1):
                        img = render_pdf_page(pdf_bytes, p, dpi=220)
                        if img is None:
                            continue
                        locator = hf_vision_chat(hf_token, LLAVA_MODEL, img,
                                                 f"Does this page contain Table {table_n}? Answer only YES or NO.",
                                                 max_tokens=10, retries=1).strip().upper()
                        if locator.startswith("YES"):
                            found = p
                            break
                    target_page = found

                if target_page is not None:
                    img = render_pdf_page(pdf_bytes, target_page, dpi=300)
                    if img is not None:
                        table_md = runtime_extract_table_md(hf_token, img, table_n)
                        table_expl = runtime_explain_table(hf_token, img, table_md, table_n)

                        runtime_text_answer = (
                            f"### ✅ Table {table_n} (Page {target_page})\n\n"
                            f"#### Extracted (DeepSeek OCR‑2)\n{table_md}\n\n"
                            f"#### Explanation (LLaVA)\n{table_expl}"
                        )

                        runtime_chunk_to_store = f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {target_page}] {runtime_text_answer}"
                        runtime_meta = {
                            "source": "runtime_ocr",
                            "filename": pdf_name,
                            "chunk_type": "table_runtime",
                            "table_number": table_n,
                            "page": target_page,
                            "created_at": datetime.utcnow().isoformat(),
                        }

        if fig_n is not None and pdf_bytes and hf_token:
            fig_missing = not has_specific_figure(context_blocks, fig_n)
            if fig_missing:
                target_page = page_n
                if target_page is not None:
                    img = render_pdf_page(pdf_bytes, target_page, dpi=260)
                    if img is not None:
                        fig_text = runtime_extract_figure(hf_token, img, fig_n)
                        fig_expl = runtime_explain_figure(hf_token, img, fig_text, fig_n)

                        runtime_text_answer = (
                            f"### ✅ Figure {fig_n} (Page {target_page})\n\n"
                            f"#### Extracted (DeepSeek OCR‑2)\n{fig_text}\n\n"
                            f"#### Explanation (LLaVA)\n{fig_expl}"
                        )

                        runtime_chunk_to_store = f"[FIGURE_RUNTIME][FIGURE {fig_n}][PAGE {target_page}] {runtime_text_answer}"
                        runtime_meta = {
                            "source": "runtime_ocr",
                            "filename": pdf_name,
                            "chunk_type": "figure_runtime",
                            "figure_number": fig_n,
                            "page": target_page,
                            "created_at": datetime.utcnow().isoformat(),
                        }

        # 3) If runtime produced a complete answer, show it directly (prevents truncation)
        if runtime_text_answer:
            try:
                st.session_state.vectorstore.add_texts([runtime_chunk_to_store], metadatas=[runtime_meta])
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
            except Exception:
                pass

            st.session_state.chat_history.append({"role": "assistant", "content": runtime_text_answer})
            append_message(SESSION_ID, "assistant", runtime_text_answer)

            # update follow-up memory
            st.session_state.last_context_blocks = ["[S0] " + runtime_chunk_to_store]
            st.session_state.last_retrieved_docs = []
            st.session_state.last_ref = ref or st.session_state.last_ref
            st.rerun()

        # 4) Otherwise, normal grounded RAG
        if not context_blocks:
            answer = "Not found in the knowledge base. Tip: include page number like 'Table 3 on page 7'."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.session_state.last_retrieved_docs = []
            st.session_state.last_context_blocks = []
            st.session_state.last_ref = ref
            st.rerun()

        grounded_msgs = [
            {"role": "system",
             "content": "You are a precise assistant. Answer using ONLY the provided snippets. Cite [S1], [S2]."},
            {"role": "user",
             "content": f"Snippets:\n{'\n\n'.join(context_blocks)}\n\nQuestion: {user_question}\nAnswer:"}
        ]
        result = llm_chat_with_fallback(
            model_id=model_id,
            messages=grounded_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            hf_token=hf_token,
            groq_key=groq_key,
            debug=st.session_state.debug_raw,
        )
        answer = normalize_text(result["content"]) or "No answer returned."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)

        st.session_state.last_retrieved_docs = last_snips[:8]
        st.session_state.last_context_blocks = context_blocks
        st.session_state.last_ref = ref or st.session_state.last_ref
        st.rerun()
