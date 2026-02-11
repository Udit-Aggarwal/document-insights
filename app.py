# app.py
"""
Streamlit Document Summarization & Q&A (RAG, Conversational Memory)

Primary LLM route:
  Hugging Face Router -> Groq provider (OpenAI-compatible chat completions)

Fallback:
  Groq Direct API (OpenAI-compatible chat completions)

Requirements:
  pip install streamlit PyPDF2 langchain-text-splitters langchain-huggingface langchain-community faiss-cpu openai
  pip install sentence-transformers transformers torch

Secrets / env:
  HUGGINGFACE_API_TOKEN  (required for HF Router)
  GROQ_API_KEY           (recommended for Groq fallback)

Notes:
- FAISS for retrieval, MiniLM embeddings locally.
- Chat-completions to avoid gen vs chat mismatches.
- Conversational Q&A with multi-turn memory and robust meta-intent handling (no LLM calls for meta).
"""

import os
import re
import traceback
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Document Q&A (HF→Groq + Groq fallback)", layout="wide")
st.title("📄 Document Summarization & Q&A (HF→Groq + Groq fallback)")


# ----------------------------
# Session state defaults
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "text" not in st.session_state:
    st.session_state.text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "debug_raw" not in st.session_state:
    st.session_state.debug_raw = False
# Chat history for multi-turn memory (list of dicts with "role" and "content")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Store last turn's retrieved snippets (display below last answer)
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []


# ----------------------------
# Helpers
# ----------------------------
def get_secret(name: str, default=None):
    """Fetch from env or Streamlit secrets."""
    return os.getenv(name) or st.secrets.get(name, default)


def hf_routed_model(model_id: str) -> str:
    """Force HF Router to use Groq provider unless already suffixed."""
    if ":" in model_id:
        return model_id
    return f"{model_id}:groq"


def normalize_text(s: str) -> str:
    return (s or "").strip()


def openai_chat(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    """Minimal OpenAI-compatible chat call."""
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
    hf_token: str | None,
    groq_key: str | None,
    debug: bool = False,
):
    """
    Primary: HF Router -> Groq provider (https://router.huggingface.co/v1)
    Fallback: Groq direct (https://api.groq.com/openai/v1)
    """
    result = {"content": "", "primary_used": False, "raw": None,
              "error_primary": None, "error_fallback": None}

    # --- Primary (HF Router)
    if not hf_token:
        result["error_primary"] = "Missing HUGGINGFACE_API_TOKEN"
    else:
        try:
            hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            routed = hf_routed_model(model_id)
            content, raw = openai_chat(
                hf_client, routed, messages, temperature=temperature, max_tokens=max_tokens
            )
            if normalize_text(content):
                result.update({"content": content, "primary_used": True, "raw": raw})
                return result
            else:
                result["error_primary"] = "Primary returned empty content."
        except Exception as e:
            result["error_primary"] = f"{type(e).__name__}: {e}"
            if debug:
                st.write("Primary exception traceback:")
                st.code(traceback.format_exc())

    # --- Fallback (Groq direct)
    if not groq_key:
        result["error_fallback"] = "Missing GROQ_API_KEY (fallback not available)"
        return result

    try:
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        content, raw = openai_chat(
            groq_client, model_id, messages, temperature=temperature, max_tokens=max_tokens
        )
        result.update({"content": content, "raw": raw})
        return result
    except Exception as e:
        result["error_fallback"] = f"{type(e).__name__}: {e}"
        if debug:
            st.write("Fallback exception traceback:")
            st.code(traceback.format_exc())
        return result


def build_qa_prompt_with_history(
    history: list[dict],
    context_blocks: list[str],
    question: str,
    max_history_turns: int = 8,
) -> list[dict]:
    """
    History-aware prompt:
    - Strict system message preferring snippets; allows chat-history only for meta questions
    - Truncated prior chat history (user/assistant turns)
    - Current user turn with tagged snippets [S1], [S2], ... and question
    """
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(msgs) > max_history_turns * 2:
        msgs = msgs[-max_history_turns * 2 :]

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Prefer answering using the provided document snippets. "
                "If the user asks about the conversation itself (e.g., last question/answer), use the chat history. "
                "If the answer is not present in snippets for document-grounded queries, say: 'Not found in the document.' "
                "When you use a snippet, cite it like [S1], [S2]."
            ),
        }
    ]
    messages.extend(msgs)

    ctx = "\n\n".join(context_blocks)
    messages.append(
        {
            "role": "user",
            "content": f"Snippets:\n{ctx}\n\nQuestion: {question}\nAnswer:",
        }
    )
    return messages


def build_summary_prompt(text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Summarize faithfully and avoid adding facts.\n"
                "Return 5-8 bullet points, concise and information-dense."
            ),
        },
        {"role": "user", "content": f"Summarize the following document:\n\n{text}"},
    ]


# ----------------------------
# Robust meta intent & safe history retrieval (NO LLM CALL)
# ----------------------------
_META_PATTERNS = {
    "last_user_q": [
        r"\b(last|previous)\s+(question|query)\b",
        r"\bwhat\s+did\s+i\s+ask\b",
        r"\bmy\s+last\s+question\b",
    ],
    "penultimate_user_q": [
        r"\b(before|prior)\s+to\s+that\b",
        r"\bquestion\s+before\s+that\b",
        r"\bprevious\s+to\s+that\b",
    ],
    "last_assistant_a": [
        r"\b(last|previous)\s+(answer|response|reply)\b",
        r"\bwhat\s+did\s+you\s+say\s+last\b",
    ],
    "penultimate_assistant_a": [
        r"\b(answer|response)\s+before\s+that\b",
        r"\bprevious\s+answer\s+to\s+that\b",
    ],
    "list_user_qs": [
        r"\b(list|show)\s+(my\s+)?(recent\s+)?(questions|queries)\b",
    ],
    "summarize_chat": [
        r"\b(summarize|recap)\s+(our|the)\s+(chat|conversation)\b",
        r"\bwhat\s+did\s+we\s+discuss\b",
    ],
}

def _match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)

def detect_meta_intent(q: str) -> str:
    """Return an intent key or 'none'."""
    ql = (q or "").strip()
    if not ql:
        return "none"
    for intent, pats in _META_PATTERNS.items():
        if _match_any(pats, ql):
            return intent
    return "none"

def _last_of_role(msgs: list[dict], role: str):
    for m in reversed(msgs):
        if m.get("role") == role:
            return m["content"]
    return None

def _nth_from_end_of_role(msgs: list[dict], role: str, n: int):
    """n=1 => last, n=2 => penultimate"""
    filtered = [m["content"] for m in msgs if m.get("role") == role]
    return filtered[-n] if len(filtered) >= n else None

def answer_meta_from_history(intent: str, history: list[dict]) -> str:
    """
    Compute deterministic answers from chat_history WITHOUT asking the LLM.
    Excludes the *current* user message by default.
    """
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    if not msgs:
        return "We haven't chatted yet."

    if intent == "last_user_q":
        # Exclude current user message (meta-ask) then get last user
        msgs_excl_current = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else msgs
        val = _last_of_role(msgs_excl_current, "user")
        return f"Your last question was: {val}" if val else "I couldn't find your last question."

    if intent == "penultimate_user_q":
        # n=2 from end, excluding this turn
        msgs_excl_current = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else msgs
        val = _nth_from_end_of_role(msgs_excl_current, "user", 2)
        return f"The question before that was: {val}" if val else "I couldn't find the question before that."

    if intent == "last_assistant_a":
        val = _last_of_role(msgs, "assistant")
        return f"My last answer was: {val}" if val else "I couldn't find my last answer."

    if intent == "penultimate_assistant_a":
        val = _nth_from_end_of_role(msgs, "assistant", 2)
        return f"The answer before that was: {val}" if val else "I couldn't find the answer before that."

    if intent == "list_user_qs":
        # List last 5 user questions (excluding this meta-ask, if any)
        msgs_excl_current = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else msgs
        qs = [m["content"] for m in msgs_excl_current if m["role"] == "user"][-5:]
        if not qs:
            return "No previous questions found."
        bullets = "\n".join(f"- {q}" for q in qs)
        return "Here are your recent questions:\n\n" + bullets

    if intent == "summarize_chat":
        last_n = msgs[-16:]
        lines = []
        for m in last_n:
            who = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content']}")
        return "Here is a brief recap of our recent conversation:\n\n" + "\n".join(lines)

    return "I'm not sure."


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Processing Options")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top K documents for retrieval", min_value=1, max_value=10, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.header("LLM Options (Primary: HF Router→Groq, Fallback: Groq Direct)")
model_id = st.sidebar.text_input("Model ID", value="openai/gpt-oss-20b")
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, step=64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, step=0.1)

st.sidebar.markdown("---")
st.session_state.debug_raw = st.sidebar.checkbox("Show raw LLM response / debug", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Chat Memory")
max_history_turns = st.sidebar.slider("Max chat turns kept in memory", 2, 20, 8, 1)
if st.sidebar.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_retrieved_docs = []
    st.toast("Chat history cleared.", icon="🧹")


# ----------------------------
# Upload area
# ----------------------------
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


# ----------------------------
# Process document button
# ----------------------------
if st.button("Process Document") and st.session_state.text:
    with st.spinner("Splitting text and creating embeddings..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )
            chunks = text_splitter.split_text(st.session_state.text)
            st.session_state.chunks = chunks

            # Local embeddings (no HF inference)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

            st.success(f"Document processed into {len(chunks)} chunks and indexed.")
        except Exception as e:
            st.error(f"Failed to create embeddings or vectorstore: {e}")


# ----------------------------
# Require vectorstore
# ----------------------------
if not st.session_state.vectorstore:
    st.info("Upload and process a document to enable Summary and Q&A.")
    st.stop()


# ----------------------------
# Tokens
# ----------------------------
hf_token = get_secret("HUGGINGFACE_API_TOKEN", None)
groq_key = get_secret("GROQ_API_KEY", None)

if not hf_token:
    st.warning("⚠️ HUGGINGFACE_API_TOKEN not found. Primary route (HF Router→Groq) will not work.")
if not groq_key:
    st.warning("ℹ️ GROQ_API_KEY not found. Fallback (Groq Direct) will not work if primary fails.")


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📝 Summary", "💬 Q&A (Chat)"])


# ----------------------------
# Summary tab
# ----------------------------
with tab1:
    st.markdown("**Generate a short summary of the document.**")

    use_chunked_summary = st.checkbox("Use chunked summary (better for long documents)", value=True)
    max_input_chars = st.slider("Max characters (non-chunked summary)", 256, 50000, 5000, step=256)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                if use_chunked_summary and st.session_state.chunks:
                    n_chunks = min(8, len(st.session_state.chunks))
                    mini_summaries = []

                    for i in range(n_chunks):
                        chunk_text = st.session_state.chunks[i]
                        messages = [
                            {"role": "system", "content": "Summarize the text in 2-3 bullet points. Avoid adding facts."},
                            {"role": "user", "content": chunk_text},
                        ]
                        out = llm_chat_with_fallback(
                            model_id=model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=min(256, max_tokens),
                            hf_token=hf_token,
                            groq_key=groq_key,
                            debug=st.session_state.debug_raw,
                        )
                        mini_summaries.append(out["content"] or "")

                    final_messages = [
                        {"role": "system", "content": "Combine the bullet points into a single 6-10 bullet executive summary. No extra facts."},
                        {"role": "user", "content": "\n\n".join(mini_summaries)},
                    ]
                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=final_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )
                else:
                    input_text = st.session_state.text[: int(max_input_chars)]
                    messages = build_summary_prompt(input_text)
                    result = llm_chat_with_fallback(
                        model_id=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        hf_token=hf_token,
                        groq_key=groq_key,
                        debug=st.session_state.debug_raw,
                    )

                if st.session_state.debug_raw:
                    st.subheader("Routing / Debug")
                    st.write(
                        {
                            "primary_used": result["primary_used"],
                            "model_primary": hf_routed_model(model_id),
                            "model_fallback": model_id,
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )
                    st.subheader("Raw LLM Response")
                    st.write(result["raw"])

                summary = normalize_text(result["content"])
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("No summary returned.")
                    st.write(
                        {
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )
            except Exception as e:
                st.error("Summarization failed.")
                st.exception(e)


# ----------------------------
# Q&A (Chat) tab — Conversational chain with memory
# ----------------------------
with tab2:
    st.markdown("**Chat with the document — multi‑turn Q&A with memory.**")

    # 1) Render prior conversation (above the input)
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    # 2) Show last turn's retrieved snippets (if any)
    if st.session_state.last_retrieved_docs:
        with st.expander("Retrieved snippets (last turn)"):
            for i, txt in enumerate(st.session_state.last_retrieved_docs, start=1):
                st.markdown(f"**Snippet {i}**")
                st.write(txt[:1500])

    # 3) Chat input pinned at the bottom
    user_question = st.chat_input("Ask a question about the document")
    if user_question:
        # Append current user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Route: meta (chat-about-chat) vs RAG
        intent = detect_meta_intent(user_question)
        if intent != "none":
            # Deterministic, zero-hallucination meta answer (uses history, excludes current turn)
            answer = answer_meta_from_history(intent, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.last_retrieved_docs = []  # no snippets for meta answers
            st.rerun()

        # Normal RAG path (document-grounded)
        with st.spinner("Retrieving and answering..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(user_question, k=int(top_k))
                if not docs:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": "No relevant documents found."}
                    )
                    st.session_state.last_retrieved_docs = []
                    st.rerun()

                # Tag snippets for this turn
                context_blocks = []
                for i, d in enumerate(docs, start=1):
                    snippet = d.page_content.strip()
                    context_blocks.append(f"[S{i}] {snippet}")

                # Build history-aware messages
                messages = build_qa_prompt_with_history(
                    history=st.session_state.chat_history,
                    context_blocks=context_blocks,
                    question=user_question,
                    max_history_turns=max_history_turns,
                )

                # Call LLM with routing & fallback
                result = llm_chat_with_fallback(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    hf_token=hf_token,
                    groq_key=groq_key,
                    debug=st.session_state.debug_raw,
                )

                if st.session_state.debug_raw:
                    st.subheader("Routing / Debug")
                    st.write(
                        {
                            "primary_used": result["primary_used"],
                            "model_primary": hf_routed_model(model_id),
                            "model_fallback": model_id,
                            "error_primary": result["error_primary"],
                            "error_fallback": result["error_fallback"],
                        }
                    )
                    st.subheader("Raw LLM Response")
                    st.write(result["raw"])

                # Show & store assistant reply
                answer = normalize_text(result["content"]) or "No answer returned."
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_retrieved_docs = [d.page_content for d in docs]

                # Re-render so input stays at the bottom
                st.rerun()

            except Exception:
                st.session_state.chat_history.append({"role": "assistant", "content": "Q&A failed."})
                st.session_state.last_retrieved_docs = []
                st.rerun()


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Notes • FAISS for vector search • HuggingFaceEmbeddings locally • Chat-completions via "
    "HF Router→Groq with Groq Direct fallback • Robust conversational memory & meta-intent handling."
)
