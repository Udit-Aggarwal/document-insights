# app.py
"""
Hardened Unified App (Backend Fixes + Better UX)
- Keeps original flow: Upload -> Process Document -> Summary + Q&A
- KB: Qdrant Cloud + FAISS fallback
- Text LLM: HF Router->Groq primary + Groq direct fallback
- Vision: DeepSeek-OCR-2 (extract) + LLaVA v1.6 34B (explain)
- Fixes:
  1) "Not found for everything" when KB not truly indexed
  2) Table 3 bug: table-specific runtime fallback (not blocked by Table 9 snippets)
  3) No S1/S2: cites (Page X) and shows sources panel with page preview
  4) Completeness: continuation for truncated outputs (vision + text)
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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Document Summarization & Q&A", layout="wide")
st.title("📄 Document Summarization & Q&A (DeepSeek OCR2 + LLaVA + Groq routing)")


# -----------------------------
# Local FAISS persistence
# -----------------------------
FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)

def save_faiss(vectorstore: FAISS, path: Path = FAISS_DIR):
    vectorstore.save_local(str(path))

def load_faiss(embeddings, path: Path = FAISS_DIR) -> Optional[FAISS]:
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return None


# -----------------------------
# SQLite chat history
# -----------------------------
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


# -----------------------------
# Session state
# -----------------------------
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("vectorstore", None)
ss_init("kb_ready", False)
ss_init("kb_count_hint", 0)

ss_init("text", "")
ss_init("chunks", [])
ss_init("page_texts", [])
