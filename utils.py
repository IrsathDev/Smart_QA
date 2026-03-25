# utils.py
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import requests

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

# -------- Load Files --------
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"text": f.read(), "page": None}]

def load_pdf(file_path):
    reader = PdfReader(file_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chunks.append({"text": text, "page": i+1})
    return chunks

def chunk_text(pages, chunk_size=500):
    """
    pages: list of dicts [{"text":..., "page":...}]
    returns list of dicts [{"chunk":..., "page":...}]
    """
    chunks = []
    for page in pages:
        text = page["text"]
        page_number = page["page"]
        for i in range(0, len(text), chunk_size):
            chunks.append({"chunk": text[i:i+chunk_size], "page": page_number})
    return chunks

# -------- Keyword Retrieval --------
def get_relevant_chunks(question, chunks, top_k=3):
    words = question.lower().split()
    scored = []

    for chunk in chunks:
        score = sum(word in chunk.lower() for word in words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]] if scored else chunks[:top_k]

# -------- NVIDIA Chat --------
def ask_nvidia(prompt, chat_history=[]):
    messages = chat_history + [{"role": "user", "content": prompt}]

    payload = {
        "model": "meta/llama3-70b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=HEADERS,
        json=payload
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]



from PyPDF2 import PdfReader

