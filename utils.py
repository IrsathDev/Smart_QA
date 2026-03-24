# utils.py
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import requests

load_dotenv()

# NVIDIA KIMI API key and base URL
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

# ---------------- File Loading ----------------
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text.append(f"[Page {i+1}] {page_text}")
    return "\n".join(text)

# ---------------- Text Chunking ----------------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- Keyword-Based Chunk Retrieval ----------------
def get_relevant_chunks(question, chunks, top_k=3):
    question_words = question.lower().split()
    relevant = [c for c in chunks if any(w in c.lower() for w in question_words)]
    return relevant[:top_k] if relevant else chunks[:top_k]

# ---------------- NVIDIA LLM Chat ----------------
def ask_nvidia(prompt, chat_history=[]):
    messages = chat_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": "meta/llama3-70b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"]
    return reply