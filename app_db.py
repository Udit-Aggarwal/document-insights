# app_db.py
"""
Document Summarization & Q&A (RAG + Conversational Memory)
KB Backends:
  1) Qdrant Cloud (recommended, uses API key)
  2) FAISS Local (fallback)

LLM Routing:
  Primary: Hugging Face Router -> Groq provider (OpenAI-compatible chat completions)
  Fallback: Groq Direct API (OpenAI-compatible chat completions)

Environment / Secrets:
  HUGGINGFACE_API_TOKEN
  GROQ_API_KEY

Qdrant (recommended KB):
  QDRANT_URL            e.g. https://xxxxx.us-east.aws.cloud.qdrant.io:6333
  QDRANT_API_KEY        (created in Qdrant Cloud console)
  QDRANT_COLLECTION     default: doc_kb

Notes:
- Uses sentence-transformers/all-MiniLM-L6-v2 embeddings (dimension=384)
- Qdrant collection is created automatically if missing
- Chat history is persisted in SQLite for conversation memory
"""

import os
import re
import sqlite3
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI

# --- Qdrant KB integration ---
# Requires: pip install langchain-qdrant qdrant-client
from langchain_qdrant import QdrantVectorStore  # [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)[5](https://pypi.org/project/langchain-qdrant/)
from qdrant_client import QdrantClient          # [7](https://pypi.org/project/qdrant-client/)
from qdrant_client.http.models import Distance, VectorParams  # [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)


# =========================
# Page config
# =========================
st.set_page_config(page_title="Document Q&A (Qdrant KB + HF→Groq)", layout="wide")
st.title("📄 Document Summarization & Q&A (Qdrant KB + HF→Groq + Groq fallback)")


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
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

init_chat_db()
if "history_synced" not in st.session_state:
    st.session_state.chat_history = load_history(SESSION_ID)
    st.session_state.history_synced = True


# =========================
# Secrets / env helpers
# =========================
def get_secret(name: str, default=None):
    return os.getenv(name) or st.secrets.get(name, default)


# =========================
# LLM routing: HF Router -> Groq (primary), Groq direct (fallback)
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
    if not hf_token:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"
    else:
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
# Prompt builders
# =========================
def build_summary_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Summarize faithfully and avoid adding facts. Return 5-8 bullet points."},
        {"role": "user", "content": f"Summarize the following document:\n\n{text}"},
    ]

def build_qa_prompt_with_history(history, context_blocks, question, max_history_turns=8):
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]

    messages = [{
        "role": "system",
        "content": (
            "You are a precise assistant. Prefer answering using the provided snippets. "
            "If the user asks about the conversation itself (e.g., last question/answer), use the chat history. "
            "If the answer is not present in snippets for document-grounded queries, say: 'Not found in the knowledge base.' "
            "When you use a snippet, cite it like [S1], [S2]."
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
    "last_assistant_a": [r"\b(last|previous)\s+(answer|response|reply)\b", r"\bwhat\s+did\s+you\s+say\s+last\b"],
    "list_user_qs": [r"\b(list|show)\s+(my\s+)?(recent\s+)?(questions|queries)\b"],
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

    if intent == "last_assistant_a":
        for m in reversed(msgs):
            if m["role"] == "assistant":
                return f"My last answer was: {m['content']}"
        return "I couldn't find my last answer."

    if intent == "list_user_qs":
        qs = [m["content"] for m in msgs_excl_current if m["role"] == "user"][-5:]
        if not qs:
            return "No previous questions found."
        return "Here are your recent questions:\n\n" + "\n".join(f"- {q}" for q in qs)

    if intent == "summarize_chat":
        last_n = msgs[-16:]
        lines = []
        for m in last_n:
            who = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content']}")
        return "Here is a brief recap of our recent conversation:\n\n" + "\n".join(lines)

    return "I'm not sure."


# =========================
# KB: Qdrant Vector Store initialization
# =========================
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

def init_qdrant_vectorstore() -> Tuple[Optional[QdrantVectorStore], Optional[str]]:
    """
    Qdrant is integrated with LangChain via langchain-qdrant. [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)[5](https://pypi.org/project/langchain-qdrant/)
    Qdrant Cloud supports API key auth; you create keys in the Cloud console. [4](https://qdrant.tech/documentation/cloud/authentication/)
    """
    url = get_secret("QDRANT_URL", None)
    api_key = get_secret("QDRANT_API_KEY", None)
    collection_name = get_secret("QDRANT_COLLECTION", "doc_kb")

    if not url or not api_key:
        return None, "Missing QDRANT_URL or QDRANT_API_KEY"

    try:
        client = QdrantClient(url=url, api_key=api_key)  # [7](https://pypi.org/project/qdrant-client/)

        # Ensure collection exists with correct vector params
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )  # [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)

        vs = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=EMBEDDINGS,
        )  # [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)

        return vs, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 50)
top_k = st.sidebar.number_input("Top K documents for retrieval", 1, 10, 4, 1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
st.session_state.debug_raw = st.sidebar.checkbox("Debug raw responses", value=False)

st.sidebar.markdown("---")
st.sidebar.header("KB Backend")
kb_backend = st.sidebar.selectbox(
    "Knowledge Base storage",
    ["Qdrant Cloud (API key)", "FAISS (Local fallback)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Chat Memory")
max_history_turns = st.sidebar.slider("Max chat turns kept", 2, 20, 8, 1)
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []
    clear_history(SESSION_ID)
    st.toast("Chat history cleared.", icon="🧹")


# =========================
# Tokens (HF + Groq)
# =========================
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)
if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN missing. Primary route will not work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY missing. Fallback won't work if primary fails.")


# =========================
# Load KB vectorstore based on backend selection
# =========================
def ensure_vectorstore():
    if kb_backend.startswith("Qdrant"):
        if not isinstance(st.session_state.vectorstore, QdrantVectorStore):
            vs, err = init_qdrant_vectorstore()
            if err:
                st.warning(f"Qdrant KB not available: {err}. Falling back to FAISS.")
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
        # FAISS local
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
# Upload + Process Document (adds to KB)
# =========================
uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
            st.session_state.text = "\n\n".join(pages)
        else:
            st.session_state.text = uploaded_file.read().decode("utf-8", errors="replace")
        st.success(f"Document loaded: {len(st.session_state.text)} characters")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if st.button("Process Document") and st.session_state.text:
    with st.spinner("Chunking + embedding + storing in KB..."):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
            )
            chunks = splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            meta_base = {
                "source": "upload",
                "filename": getattr(uploaded_file, "name", "unknown"),
                "filetype": getattr(uploaded_file, "type", "unknown"),
                "uploaded_at": datetime.utcnow().isoformat(),
            }
            metadatas = [meta_base for _ in chunks]

            ensure_vectorstore()
            if st.session_state.vectorstore is None:
                # Create FAISS if Qdrant not configured
                st.session_state.vectorstore = FAISS.from_texts(chunks, EMBEDDINGS, metadatas=metadatas)
                save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(chunks)} chunks in FAISS local KB.")
            else:
                st.session_state.vectorstore.add_texts(chunks, metadatas=metadatas)
                if isinstance(st.session_state.vectorstore, FAISS):
                    save_faiss(st.session_state.vectorstore)
                st.session_state.kb_ready = True
                st.success(f"Stored {len(chunks)} chunks in KB.")
        except Exception as e:
            st.error(f"Failed to process document: {e}")


# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# =========================
# Summary tab
# =========================
with tab1:
    st.markdown("**Generate a summary of the uploaded document.**")
    use_chunked_summary = st.checkbox("Use chunked summary (better for long documents)", value=True)
    max_input_chars = st.slider("Max characters (non-chunked summary)", 256, 50000, 5000, 256)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            try:
                if use_chunked_summary and st.session_state.chunks:
                    n_chunks = min(8, len(st.session_state.chunks))
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
                        {"role": "system", "content": "Combine into 6-10 bullet executive summary. No extra facts."},
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
# Q&A tab — chat over KB (Qdrant/FAISS)
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

        # RAG retrieval
        with st.spinner("Retrieving and answering..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(user_question, k=int(top_k))
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
    "Qdrant integrates with LangChain via langchain-qdrant and supports Qdrant Cloud with API key authentication. "
    "API keys are created in the Qdrant Cloud console. [2](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)[4](https://qdrant.tech/documentation/cloud/authentication/)"
)
