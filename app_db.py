# app.py
"""
Unified Robust RAG + Visual Grounding:
- Text chat: HF Router -> Groq primary + Groq Direct fallback
- KB: Qdrant (preferred) + FAISS fallback
- Vision stack (HF token):
    * DeepSeek-OCR-2: extraction / OCR / markdown table conversion (structured) [1](https://www.geeksforgeeks.org/artificial-intelligence/how-to-get-a-groq-api-key/)[2](https://deepwiki.com/groq/groq-api-cookbook/1.1-getting-started)
    * LLaVA v1.6 34B: explanation / reasoning about table/figure meaning [3](https://portkey.ai/models/groq/llama-3.1-70b-versatile)
- Runtime fallback:
    If table/figure not found in KB:
      -> render PDF page
      -> DeepSeek extracts table (markdown)
      -> LLaVA explains it
      -> auto-insert into KB
      -> answer immediately (one-go)

Notes:
- Base64 image payload to HF can trigger HTTP 413 Payload Too Large; we downscale/compress. [4](https://paravisionlab.co.in/text-summarizer/)
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
# Page config
# =========================
st.set_page_config(page_title="Unified Doc Q&A (DeepSeek OCR2 + LLaVA)", layout="wide")
st.title("📄 Unified Document Q&A (DeepSeek OCR‑2 + LLaVA v1.6 34B) — robust tables/figures")


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
# SQLite chat memory
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_context_blocks" not in st.session_state:
    st.session_state.last_context_blocks = []
if "last_ref" not in st.session_state:
    st.session_state.last_ref = None

# store PDF bytes for runtime fallback
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
# Text LLM routing: HF Router -> Groq (primary) + Groq direct fallback
# =========================
def hf_routed_model(model_id: str) -> str:
    return model_id if ":" in model_id else f"{model_id}:groq"

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

    # Primary: HF Router -> Groq
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
# Vision models via HF Inference (OpenAI compatible)
# =========================
HF_INFER_BASE = "https://api-inference.huggingface.co/v1/"

DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR-2"      # OCR/table markdown conversion [1](https://www.geeksforgeeks.org/artificial-intelligence/how-to-get-a-groq-api-key/)[2](https://deepwiki.com/groq/groq-api-cookbook/1.1-getting-started)
LLAVA_MODEL = "llava-hf/llava-v1.6-34b-hf"            # visual reasoning/explanation [3](https://portkey.ai/models/groq/llama-3.1-70b-versatile)

def image_to_data_url(img: Image.Image, max_bytes: int = 650_000) -> str:
    """
    Downscale/compress to reduce risk of 413 Payload Too Large with base64 images. [4](https://paravisionlab.co.in/text-summarizer/)
    """
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

def hf_vision_chat(hf_token: str, model: str, img: Image.Image, prompt: str, max_tokens: int = 1200, retries: int = 2) -> str:
    """
    Calls HF OpenAI-compatible inference endpoint with image_url + text content blocks. [5](https://console.groq.com/docs/text-chat)
    """
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
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content if resp and resp.choices else ""
            return (out or "").strip()
        except Exception as e:
            last_err = str(e)
            # handle payload too large by compressing harder [4](https://paravisionlab.co.in/text-summarizer/)
            if "413" in last_err or "Payload Too Large" in last_err:
                data_url = image_to_data_url(img, max_bytes=450_000)
                messages[0]["content"][0]["image_url"]["url"] = data_url
            time.sleep(1.0)

    return ""


# =========================
# Parse references (table/figure/page)
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
# Render PDF page to image
# =========================
def render_pdf_page(pdf_bytes: bytes, page_num_1based: int, dpi: int = 260) -> Optional[Image.Image]:
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
# Retrieval helpers (visual prioritized)
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

    # De-dup
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
# Prompt builders for grounded Q&A
# =========================
def build_grounded_prompt(context_blocks: List[str], question: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer using ONLY the provided snippets. "
                "If the answer is not present in snippets, say: 'Not found in the knowledge base.' "
                "Cite snippet IDs like [S1], [S2]."
            ),
        },
        {"role": "user", "content": f"Snippets:\n{'\n\n'.join(context_blocks)}\n\nQuestion: {question}\nAnswer:"},
    ]


# =========================
# Sidebar
# =========================
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox("Knowledge Base storage", ["Qdrant Cloud (API key)", "FAISS (Local fallback)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Query Options")
top_k = st.sidebar.number_input("Top K retrieval", 1, 12, 6, 1)
scan_pages_limit = st.sidebar.number_input("Max pages to scan if page not specified", 1, 80, 25, 1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID (text LLM)", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
st.session_state.debug_raw = st.sidebar.checkbox("Debug", value=False)

st.sidebar.markdown("---")
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
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. DeepSeek/LLaVA runtime extraction will not work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. Groq fallback won't work if HF Router fails.")


# =========================
# Ensure vectorstore ready
# =========================
ensure_vectorstore(kb_backend)


# =========================
# Upload PDF (required for runtime extraction)
# =========================
uploaded_file = st.file_uploader("Upload PDF (required for table/figure runtime extraction)", type=["pdf"])
if uploaded_file:
    st.session_state.last_pdf_bytes = uploaded_file.getvalue()
    st.session_state.last_pdf_name = uploaded_file.name
    st.success(f"PDF loaded in session: {uploaded_file.name}")


# =========================
# Tabs
# =========================
tab_chat = st.tabs(["💬 Q&A (Chat)"])[0]


# =========================
# Runtime extractors (dual model: DeepSeek for OCR, LLaVA for explain)
# =========================
def runtime_extract_table(hf_token: str, img: Image.Image, table_n: int) -> str:
    """
    Use DeepSeek OCR2 to convert to markdown / extract table. [1](https://www.geeksforgeeks.org/artificial-intelligence/how-to-get-a-groq-api-key/)[2](https://deepwiki.com/groq/groq-api-cookbook/1.1-getting-started)
    """
    # DeepSeek prompt patterns from model guidance [1](https://www.geeksforgeeks.org/artificial-intelligence/how-to-get-a-groq-api-key/)[2](https://deepwiki.com/groq/groq-api-cookbook/1.1-getting-started)
    prompt = (
        f"<image>\n<|grounding|>Convert the document to markdown.\n"
        f"Focus on Table {table_n}. If a table exists, output it as a markdown table."
    )
    return hf_vision_chat(hf_token, DEEPSEEK_OCR_MODEL, img, prompt, max_tokens=1600, retries=2)

def runtime_explain_table(hf_token: str, img: Image.Image, extracted_markdown: str, table_n: int) -> str:
    """
    Use LLaVA to explain meaning / patterns of the table, grounded on extracted markdown. [3](https://portkey.ai/models/groq/llama-3.1-70b-versatile)
    """
    prompt = (
        f"This image contains Table {table_n}. "
        "Explain the table clearly.\n"
        "Use the extracted table text below as the ground truth. If something is unclear, say so.\n\n"
        f"Extracted Table (markdown):\n{extracted_markdown}\n\n"
        "Now explain: columns meaning, key values, comparisons, patterns, and main takeaway."
    )
    return hf_vision_chat(hf_token, LLAVA_MODEL, img, prompt, max_tokens=900, retries=2)

def runtime_extract_figure(hf_token: str, img: Image.Image, fig_n: int) -> str:
    """
    Use DeepSeek OCR2 'Parse the figure' style, then we can also ask LLaVA to explain. [2](https://deepwiki.com/groq/groq-api-cookbook/1.1-getting-started)[1](https://www.geeksforgeeks.org/artificial-intelligence/how-to-get-a-groq-api-key/)
    """
    prompt = f"<image>\nParse the figure. Figure {fig_n}. Describe axes/labels and key elements."
    return hf_vision_chat(hf_token, DEEPSEEK_OCR_MODEL, img, prompt, max_tokens=1200, retries=2)

def runtime_explain_figure(hf_token: str, img: Image.Image, fig_n: int, extracted: str) -> str:
    prompt = (
        f"This image contains Figure {fig_n}. Explain it clearly.\n"
        "Use the extracted details below as grounding:\n\n"
        f"{extracted}\n\n"
        "Now provide: what it shows, axes/legend, key trends, and takeaway."
    )
    return hf_vision_chat(hf_token, LLAVA_MODEL, img, prompt, max_tokens=900, retries=2)


# =========================
# Q&A loop (RAG + runtime fallback)
# =========================
with tab_chat:
    st.markdown("**Ask about your PDF. For one-go accuracy use: “Explain Table 3 on page 7”**")

    # render chat history
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
            answer = "Knowledge base not ready. Configure Qdrant or FAISS."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.rerun()

        # parse refs
        table_n = parse_table_number(user_question)
        fig_n = parse_figure_number(user_question)
        page_n = parse_page_number(user_question)
        ref = extract_ref(user_question)
        prefer_visual = is_visual_question(user_question)

        reuse_prior = (ref is not None and ref == st.session_state.last_ref and len(st.session_state.last_context_blocks) > 0)

        # 1) RAG retrieval first
        k = max(int(top_k), 10) if prefer_visual else int(top_k)
        docs = retrieve_docs(st.session_state.vectorstore, user_question, k=k, prefer_visual=prefer_visual)

        context_blocks: List[str] = []
        last_snips: List[str] = []

        for i, d in enumerate(docs, start=1):
            snippet = getattr(d, "page_content", "").strip()
            if snippet:
                context_blocks.append(f"[S{i}] {snippet}")
                last_snips.append(snippet)

        # merge prior context
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
        pdf_name  = st.session_state.last_pdf_name or "uploaded.pdf"

        # helper
        def kb_has_table(blocks: List[str]) -> bool:
            return any(("TABLE_" in b or "table_runtime" in b.lower()) for b in blocks)

        def kb_has_figure(blocks: List[str]) -> bool:
            return any(("FIGURE_" in b or "figure_runtime" in b.lower()) for b in blocks)

        # 2) Runtime fallback if missing
        runtime_chunks_to_add: List[Tuple[str, Dict]] = []

        # ---- Table runtime fallback (DeepSeek extract -> LLaVA explain) ----
        if table_n is not None and (not kb_has_table(context_blocks)) and pdf_bytes and hf_token:
            # If page provided => one-go deterministic extraction
            if page_n is not None:
                img = render_pdf_page(pdf_bytes, page_n, dpi=280)
                if img is not None:
                    extracted_md = runtime_extract_table(hf_token, img, table_n)
                    explained = runtime_explain_table(hf_token, img, extracted_md, table_n)

                    runtime_text = (
                        f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {page_n}]\n\n"
                        f"### Extracted (DeepSeek OCR2)\n{extracted_md}\n\n"
                        f"### Explanation (LLaVA)\n{explained}"
                    )
                    context_blocks.insert(0, f"[S0] {runtime_text}")

                    runtime_chunks_to_add.append((
                        runtime_text,
                        {
                            "source": "runtime_ocr",
                            "filename": pdf_name,
                            "chunk_type": "table_runtime",
                            "table_number": table_n,
                            "page": page_n,
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    ))
            else:
                # No page specified: scan first N pages for "Table N" using a cheap yes/no query to LLaVA
                # then do the deterministic extraction+explain on found page.
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    scan_pages = min(int(scan_pages_limit), doc.page_count)
                    doc.close()
                except Exception:
                    scan_pages = int(scan_pages_limit)

                found_page = None
                for p in range(1, scan_pages + 1):
                    img = render_pdf_page(pdf_bytes, p, dpi=220)
                    if img is None:
                        continue
                    # quick locator using LLaVA
                    ans = hf_vision_chat(
                        hf_token, LLAVA_MODEL, img,
                        f"Does this page contain Table {table_n}? Answer only YES or NO.",
                        max_tokens=10, retries=1
                    ).strip().upper()
                    if ans.startswith("YES"):
                        found_page = p
                        break

                if found_page is not None:
                    img = render_pdf_page(pdf_bytes, found_page, dpi=280)
                    if img is not None:
                        extracted_md = runtime_extract_table(hf_token, img, table_n)
                        explained = runtime_explain_table(hf_token, img, extracted_md, table_n)

                        runtime_text = (
                            f"[TABLE_RUNTIME][TABLE {table_n}][PAGE {found_page}]\n\n"
                            f"### Extracted (DeepSeek OCR2)\n{extracted_md}\n\n"
                            f"### Explanation (LLaVA)\n{explained}"
                        )
                        context_blocks.insert(0, f"[S0] {runtime_text}")

                        runtime_chunks_to_add.append((
                            runtime_text,
                            {
                                "source": "runtime_ocr",
                                "filename": pdf_name,
                                "chunk_type": "table_runtime",
                                "table_number": table_n,
                                "page": found_page,
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        ))

        # ---- Figure runtime fallback (DeepSeek parse -> LLaVA explain) ----
        if fig_n is not None and (not kb_has_figure(context_blocks)) and pdf_bytes and hf_token:
            if page_n is not None:
                img = render_pdf_page(pdf_bytes, page_n, dpi=260)
                if img is not None:
                    extracted = runtime_extract_figure(hf_token, img, fig_n)
                    explained = runtime_explain_figure(hf_token, img, fig_n, extracted)
                    runtime_text = (
                        f"[FIGURE_RUNTIME][FIGURE {fig_n}][PAGE {page_n}]\n\n"
                        f"### Extracted (DeepSeek OCR2)\n{extracted}\n\n"
                        f"### Explanation (LLaVA)\n{explained}"
                    )
                    context_blocks.insert(0, f"[S0] {runtime_text}")
                    runtime_chunks_to_add.append((
                        runtime_text,
                        {
                            "source": "runtime_ocr",
                            "filename": pdf_name,
                            "chunk_type": "figure_runtime",
                            "figure_number": fig_n,
                            "page": page_n,
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    ))

        # 3) Auto-insert runtime chunks into KB (so next query is instant)
        if runtime_chunks_to_add:
            try:
                texts = [t for (t, m) in runtime_chunks_to_add]
                metas = [m for (t, m) in runtime_chunks_to_add]
                st.session_state.vectorstore.add_texts(texts, metadatas=metas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
            except Exception:
                pass

        # 4) If still no context, fail gracefully
        if not context_blocks:
            if not pdf_bytes:
                answer = "Not found in KB and no PDF is loaded for runtime extraction. Upload the PDF and ask again."
            elif not hf_token:
                answer = "Not found in KB and HUGGINGFACE_API_TOKEN is missing, so runtime extraction can't run."
            else:
                answer = "Not found in the knowledge base."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            append_message(SESSION_ID, "assistant", answer)
            st.session_state.last_retrieved_docs = []
            st.session_state.last_context_blocks = []
            st.session_state.last_ref = ref
            st.rerun()

        # 5) Ask text LLM grounded ONLY on snippets
        messages = build_grounded_prompt(context_blocks, user_question)
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

        # 6) Anti-false-not-found retry if we have snippets
        if ("not found in the knowledge base" in answer.lower() or answer.strip().lower() == "not found in the knowledge base.") and len(context_blocks) > 0:
            retry_msgs = [
                {"role": "system", "content": "You MUST answer using the snippets below. Do NOT say 'Not found' if relevant content exists."},
                {"role": "user", "content": f"Snippets:\n{'\n\n'.join(context_blocks)}\n\nQuestion: {user_question}\nAnswer:"}
            ]
            retry = llm_chat_with_fallback(
                model_id=model_id,
                messages=retry_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                hf_token=hf_token,
                groq_key=groq_key,
                debug=st.session_state.debug_raw,
            )
            answer2 = normalize_text(retry["content"])
            if answer2:
                answer = answer2

        # persist answer
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        append_message(SESSION_ID, "assistant", answer)

        # store last context for follow-ups
        st.session_state.last_retrieved_docs = last_snips[:8]
        st.session_state.last_context_blocks = context_blocks
        st.session_state.last_ref = ref or st.session_state.last_ref

        st.rerun()
