# app.py
"""
Ultimate Final: Document Summarization & Q&A (RAG + Conversational Memory)
KB Backends:
  1) Qdrant Cloud (API key)
  2) FAISS Local (fallback)

LLM Routing:
  Primary: Hugging Face Router -> Groq provider (OpenAI-compatible chat completions)
  Fallback: Groq Direct (OpenAI-compatible chat completions)

OCR / Figures:
  DeepSeek-OCR-2 called via Hugging Face token (no pytesseract).
  Uses prompt formats from model guidance: "<image>\\nFree OCR." and "<image>\\nParse the figure."

Key robustness:
- Better PDF parsing via PyMuPDF (fitz) rather than PyPDF2.
- OCR-enrichment stored into KB to support:
  - text in figures / charts / diagrams
  - text that might be lost in normal extraction (e.g., subscripts)
- Image compression/downscaling to avoid payload-too-large errors.

ENV / Secrets:
  HUGGINGFACE_API_TOKEN
  GROQ_API_KEY
  QDRANT_URL
  QDRANT_API_KEY
  QDRANT_COLLECTION (optional)
"""

import os
import re
import io
import base64
import time
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


# =========================
# Page config
# =========================
st.set_page_config(page_title="Doc Q&A (Qdrant KB + DeepSeek OCR2)", layout="wide")
st.title("📄 Document Summarization & Q&A (DeepSeek‑OCR‑2 for Figures + Robust Text)")


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
# SQLite chat memory (persistent)
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
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id
init_chat_db()
if "history_synced" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)
    st.session_state.history_synced = True


# =========================
# Env helpers
# =========================
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)


# =========================
# LLM routing: HF Router -> Groq (primary) + Groq fallback
# =========================
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

def normalize_text(s: str) -> str:
    return (s or "").strip()

def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content, resp

def llm_chat_with_fallback(
    model_id: str,
    messages,
    temperature: float,
    max_tokens: int,
    hf_token: Optional[str],
    groq_key: Optional[str],
    debug: bool = False,
):
    result = {"content": "", "primary_used": False, "raw": None,
              "error_primary": None, "error_fallback": None}

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
# DeepSeek-OCR-2 via HF token
# =========================
DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"

def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    """
    Convert PIL image to a compressed JPEG data URL.
    Keeps payload small to reduce risk of 413 Payload Too Large.
    """
    img = img.convert("RGB")

    # Downscale by max side
    max_side = 1300
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    # Compress iteratively
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

def deepseek_ocr_call(hf_token: str, img: Image.Image, prompt: str, max_tokens: int = 1200, retries: int = 2) -> str:
    """
    Calls DeepSeek-OCR-2 through HF Inference OpenAI-compatible endpoint.
    """
    if not hf_token:
        return ""

    data_url = image_to_data_url(img)
    client = OpenAI(base_url="https://api-inference.huggingface.co/v1/", api_key=hf_token)

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
                model=DEEPSEEK_OCR_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            return (out or "").strip()
        except Exception as e:
            last_err = str(e)
            # if payload too large, shrink harder and retry
            if "413" in last_err or "Payload Too Large" in last_err:
                # Shrink more aggressively next try
                data_url = image_to_data_url(img, max_bytes=450_000)
                messages[0]["content"][0]["image_url"]["url"] = data_url
            time.sleep(1.0)

    return ""


# =========================
# Superscript/subscript best-effort + OCR enrichment
# =========================
_SUP_MAP = str.maketrans({"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"})
_SUB_MAP = str.maketrans({"0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉"})

def to_sup(text: str) -> str:
    return text.translate(_SUP_MAP) if text else text

def to_sub(text: str) -> str:
    return text.translate(_SUB_MAP) if text else text

def extract_text_with_scripts(page) -> str:
    """
    Extract text from PDF page and best-effort mark super/subscripts.
    Superscripts: span flags + geometry.
    Subscripts: geometry heuristic only (OCR enrichment is what makes it robust).
    """
    d = page.get_text("dict")
    out_lines = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            line_bbox = line.get("bbox")
            spans = line.get("spans", [])
            if not spans:
                continue

            base_size = max(s.get("size", 0) for s in spans) or spans[0].get("size", 10)
            ly0, ly1 = (line_bbox[1], line_bbox[3]) if line_bbox else (None, None)

            line_text = ""
            for s in spans:
                txt = s.get("text", "")
                if not txt:
                    continue
                size = s.get("size", base_size)
                flags = s.get("flags", 0)
                sb = s.get("bbox")
                sy0, sy1 = (sb[1], sb[3]) if sb else (None, None)

                # superscript best-effort
                is_super = bool(flags & 1)
                if (not is_super) and ly0 is not None and sy1 is not None:
                    if size <= base_size * 0.80 and sy1 < (ly0 + (ly1 - ly0) * 0.55):
                        is_super = True

                # subscript heuristic
                is_sub = False
                if ly0 is not None and sy0 is not None:
                    if size <= base_size * 0.85 and sy0 > (ly0 + (ly1 - ly0) * 0.55):
                        is_sub = True

                if is_super and not is_sub:
                    conv = to_sup(txt)
                    line_text += conv if conv != txt else f"^({txt})"
                elif is_sub:
                    conv = to_sub(txt)
                    line_text += conv if conv != txt else f"_({txt})"
                else:
                    line_text += txt

            if line_text.strip():
                out_lines.append(line_text.strip())

    return "\n".join(out_lines).strip()


# =========================
# PDF ingestion (text + DeepSeek OCR2 enrichment)
# =========================
FIG_CAPTION_RE = re.compile(r"^\s*(fig\.?|figure)\s*\d+[:.\-\s]", re.IGNORECASE)

def extract_fig_captions(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if FIG_CAPTION_RE.match(ln)]

def render_page_image(page, dpi: int = 180) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def extract_pdf_artifacts_with_deepseek(
    pdf_bytes: bytes,
    filename: str,
    hf_token: str,
    do_page_ocr: bool,
    do_image_ocr: bool,
    do_figure_parse: bool,
    page_ocr_mode: str,
) -> Tuple[str, List[str]]:
    """
    page_ocr_mode:
      - 'auto': OCR only when native text small
      - 'all': OCR all pages (most robust, more API calls)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_pages_text = []
    extras = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)

        # 1) native text + best-effort scripts
        page_text = extract_text_with_scripts(page)
        if page_text:
            all_pages_text.append(f"[PAGE {pno+1}]\n{page_text}")
            for cap in extract_fig_captions(page_text):
                extras.append(f"[FIGURE_CAPTION][PAGE {pno+1}] {cap}")
        else:
            all_pages_text.append(f"[PAGE {pno+1}]\n")

        # 2) page OCR (captures vector figures + hidden scripts)
        if do_page_ocr and hf_token:
            should_ocr = True if page_ocr_mode == "all" else (len(page_text) < 60)
            if should_ocr:
                img = render_page_image(page, dpi=180)
                ocr_prompt = "<image>\nFree OCR."
                ocr = deepseek_ocr_call(hf_token, img, ocr_prompt, max_tokens=1400, retries=2)
                if ocr:
                    extras.append(f"[PAGE_OCR][PAGE {pno+1}] {ocr}")

        # 3) embedded image OCR (labels in figures)
        if do_image_ocr and hf_token:
            try:
                img_list = page.get_images(full=True)
            except Exception:
                img_list = []

            for idx, imginfo in enumerate(img_list, start=1):
                xref = imginfo[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image", b"")
                    if not img_bytes or len(img_bytes) < 1500:
                        continue
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    ocr_prompt = "<image>\nFree OCR."
                    ocr = deepseek_ocr_call(hf_token, pil_img, ocr_prompt, max_tokens=900, retries=2)
                    if ocr:
                        extras.append(f"[FIGURE_OCR][PAGE {pno+1}][IMG {idx}] {ocr}")

                    if do_figure_parse:
                        parse_prompt = "<image>\nParse the figure."
                        parsed = deepseek_ocr_call(hf_token, pil_img, parse_prompt, max_tokens=1200, retries=2)
                        if parsed:
                            extras.append(f"[FIGURE_PARSE][PAGE {pno+1}][IMG {idx}] {parsed}")

                except Exception:
                    continue

    doc.close()
    main_text = "\n\n".join(all_pages_text).strip()
    return main_text, extras


# =========================
# KB: Qdrant init
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


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 12, 6, 1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
st.session_state.debug_raw = st.sidebar.checkbox("Debug raw responses", value=False)

st.sidebar.markdown("---")
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local fallback)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("DeepSeek OCR2 Enrichment")
do_page_ocr = st.sidebar.checkbox("Page OCR (captures scripts + vector figures)", value=True)
page_ocr_mode = st.sidebar.selectbox("Page OCR mode", ["auto", "all"], index=0,
                                     help="auto = OCR only when extracted text is small; all = OCR every page.")
do_image_ocr = st.sidebar.checkbox("OCR embedded images (figures)", value=True)
do_figure_parse = st.sidebar.checkbox("Parse figures for better explanations", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Chat Memory")
max_history_turns = st.sidebar.slider("Max chat turns kept", 2, 20, 8, 1)
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.", icon="🧹")


# =========================
# Tokens
# =========================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. DeepSeek OCR2 and primary LLM route will not work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. LLM fallback won't work if primary fails.")


# =========================
# Load KB vectorstore
# =========================
def ensure_vectorstore():
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

ensure_vectorstore()


# =========================
# Prompt builders for Summary/Q&A
# =========================
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 8-12 bullet points."},
        {"role": "user", "content": f"Summarize the following content:\n\n{text}"},
    ]

def build_qa_prompt_with_history(history, context_blocks, question, max_history_turns=8):
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]

    messages = [{
        "role": "system",
        "content": (
            "You are a precise assistant. Prefer answering using the provided snippets. "
            "If the answer is not present in snippets, say: 'Not found in the knowledge base.' "
            "When you use a snippet, cite it like [S1], [S2]. "
            "If the question asks to explain a figure, use any [FIGURE_PARSE], [FIGURE_OCR], and [FIGURE_CAPTION] snippets."
        ),
    }]
    messages.extend(msgs)

    ctx = "\n\n".join(context_blocks)
    messages.append({"role": "user", "content": f"Snippets:\n{ctx}\n\nQuestion: {question}\nAnswer:"})
    return messages


# =========================
# Deterministic "chat-about-chat" (no hallucination)
# =========================
_META_PATTERNS = {
    "last_user_q": [r"\b(last|previous)\s+(question|query)\b", r"\bwhat\s+did\s+i\s+ask\b", r"\bmy\s+last\s+question\b"],
    "summarize_chat": [r"\b(summarize|recap)\s+(our|the)\s+(chat|conversation)\b", r"\bwhat\s+did\s+we\s+discuss\b"],
}

def detect_meta_intent(q: str) -> str:
    ql = (q or "").strip()
    if not ql:
        return "none"
    for intent, pats in _META_PATTERNS.items():
        if any(re.search(p, ql, flags=re.I) for p in pats):
            return intent
    return "none"

def answer_meta_from_history(intent: str, history: list[dict]) -> str:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if not msgs:
        return "We haven't chatted yet."
    msgs_excl_current = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else msgs

    if intent == "last_user_q":
        for m in reversed(msgs_excl_current):
            if m["role"] == "user":
                return f"Your last question was: {m['content']}"
        return "I couldn't find your last question."

    if intent == "summarize_chat":
        last_n = msgs[-16:]
        lines = []
        for m in last_n:
            who = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content']}")
        return "Here is a brief recap:\n\n" + "\n".join(lines)

    return "I'm not sure."


# =========================
# Upload + Process Document
# =========================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            main_text, extras = extract_pdf_artifacts_with_deepseek(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                hf_token=hf_token or "",
                do_page_ocr=bool(do_page_ocr and hf_token),
                do_image_ocr=bool(do_image_ocr and hf_token),
                do_figure_parse=bool(do_figure_parse and hf_token),
                page_ocr_mode=page_ocr_mode,
            )
            st.session_state.text = main_text
            st.session_state._extra_texts = extras
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")
            st.session_state._extra_texts = []

        st.success(f"Document loaded: {len(st.session_state.text)} characters")
        if getattr(st.session_state, "_extra_texts", []):
            st.info(f"DeepSeek OCR2 extracted {len(st.session_state._extra_texts)} extra snippets (OCR/captions/figure parse).")
        if uploaded_file.type == "application/pdf" and not hf_token:
            st.warning("DeepSeek OCR2 is disabled because HUGGINGFACE_API_TOKEN is missing.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB (incl. DeepSeek OCR2 enrichment)..."):
        try:
            ensure_vectorstore()
            splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

            # Split main text
            chunks = splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            # Additional OCR/caption/parse chunks
            extras = getattr(st.session_state, "_extra_texts", [])
            extra_chunks = []
            for t in extras:
                if len(t) > 2500:
                    extra_chunks.extend(splitter.split_text(t))
                else:
                    extra_chunks.append(t)

            all_texts = chunks + extra_chunks

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown"),
                "filetype": getattr(uploaded_file, "type", "unknown"),
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            metadatas = []
            for t in all_texts:
                m = dict(meta_base)
                if t.startswith("[PAGE_OCR]"):
                    m["chunk_type"] = "page_ocr"
                elif t.startswith("[FIGURE_OCR]"):
                    m["chunk_type"] = "figure_ocr"
                elif t.startswith("[FIGURE_PARSE]"):
                    m["chunk_type"] = "figure_parse"
                elif t.startswith("[FIGURE_CAPTION]"):
                    m["chunk_type"] = "figure_caption"
                else:
                    m["chunk_type"] = "main_text"
                metadatas.append(m)

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_texts(all_texts, EMBEDDINGS, metadatas=metadatas)
                save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_texts)} chunks in FAISS KB.")
            else:
                st.session_state.vectorstore.add_texts(all_texts, metadatas=metadatas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(all_texts)} chunks in KB (incl. DeepSeek OCR2).")

        except Exception as e:
            st.error(f"Failed to process document: {e}")
            st.exception(e)


# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# =========================
# Summary tab
# =========================
with tab1:
    st.markdown("**Generate a summary of the uploaded document.**")
    use_chunked_summary = st.checkbox("Use chunked summary (better for long docs)", value=True)
    max_input_chars = st.slider("Max characters (non-chunked summary)", 256, 50000, 8000, 256)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            try:
                if use_chunked_summary and st.session_state.chunks:
                    n_chunks = min(12, len(st.session_state.chunks))
                    mini = []
                    for i in range(n_chunks):
                        messages = [
                            {"role": "system", "content": "Summarize in 2-3 bullet points. Avoid adding facts."},
                            {"role": "user", "content": st.session_state.chunks[i]},
                        ]
                        out = llm_chat_with_fallback(
                            model_id=model_id, messages=messages, temperature=temperature,
                            max_tokens=min(256, max_tokens), hf_token=hf_token, groq_key=groq_key,
                            debug=st.session_state.debug_raw,
                        )
                        mini.append(out["content"] or "")

                    final_messages = [
                        {"role": "system", "content": "Combine into 8-12 bullet executive summary. No extra facts."},
                        {"role": "user", "content": "\n\n".join(mini)},
                    ]
                    result = llm_chat_with_fallback(
                        model_id=model_id, messages=final_messages, temperature=temperature,
                        max_tokens=max_tokens, hf_token=hf_token, groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )
                else:
                    input_text = st.session_state.text[: int(max_input_chars)]
                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=build_summary_prompt(input_text),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )

                summary = normalize_text(result["content"])
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("No summary returned.")
            except Exception as e:
                st.error("Summarization failed.")
                st.exception(e)


# =========================
# Q&A tab — chat over KB
# =========================
with tab2:
    st.markdown("**Chat with the Knowledge Base — multi‑turn memory enabled.**")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    # Show last retrieved snippets
    if st.session_state.last_retrieved_docs:
        with st.expander("Retrieved snippets (last turn)"):
            for i, txt in enumerate(st.session_state.last_retrieved_docs, start=1):
                st.markdown(f"**Snippet {i}**")
                st.write(txt[:1500])

    user_question = st.chat_input("Ask a question about the knowledge base")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        append_message(SESSION_ID, "user", user_question)

        ensure_vectorstore()
        if st.session_state.vectorstore is None:
            answer = "Knowledge base not ready. Configure Qdrant or process at least one document."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.rerun()

        # Meta questions
        intent = detect_meta_intent(user_question)
        if intent != "none":
            answer = answer_meta_from_history(intent, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.session_state.last_retrieved_docs = []
            st.rerun()

        # Increase retrieval for figure queries
        k = int(top_k)
        if re.search(r"\b(fig|figure|diagram|chart|graph|image)\b", user_question, flags=re.I):
            k = max(k, 10)

        with st.spinner("Retrieving and answering..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(user_question, k=k)
                if not docs:
                    answer = "No relevant content found in the knowledge base."
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    append_message(SESSION_ID, "assistant", answer)
                    st.session_state.last_retrieved_docs = []
                    st.rerun()

                context_blocks = []
                last_snips = []
                for i, d in enumerate(docs, start=1):
                    snippet = getattr(d, "page_content", "").strip()
                    context_blocks.append(f"[S{i}] {snippet}")
                    last_snips.append(snippet)

                messages = build_qa_prompt_with_history(
                    st.session_state.chat_history,
                    context_blocks,
                    user_question,
                    max_history_turns=max_history_turns,
                )

                result = llm_chat_with_fallback(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                    groq_key=groq_key,
                    debug=st.session_state.debug_raw,
                )

                answer = normalize_text(result["content"]) or "No answer returned."
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                append_message(SESSION_ID, "assistant", answer)
                st.session_state.last_retrieved_docs = last_snips
                st.rerun()

            except Exception:
                st.session_state.chat_history.append({"role": "assistant", "content": "Q&A failed."})
                append_message(SESSION_ID, "assistant", "Q&A failed.")
                st.session_state.last_retrieved_docs = []
                st.rerun()


st.markdown("---")
st.caption(
    "Uses DeepSeek‑OCR‑2 via Hugging Face token with prompts like '<image>\\nFree OCR.' and '<image>\\nParse the figure.' "
    "Large image payloads can trigger 413, so images are compressed before upload. "
    "Subscripts are not reliably detectable in native PDF extraction, so OCR enrichment improves robustness."
)
