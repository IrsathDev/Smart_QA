# utils.py
# Step 1 Upgrade: Added overlap chunking for better context preservation

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


# ── LOAD TEXT FILE ────────────────────────────────────────────────────────────
def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ── LOAD PDF WITH PAGE NUMBERS ────────────────────────────────────────────────
def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text.append(f"[Page {i+1}] {page_text}")
    return "\n".join(text)


# ── IMPROVED CHUNKING WITH OVERLAP ────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.
    Overlap ensures context is not lost at chunk boundaries.
    
    Example: chunk_size=500, overlap=50
    chunk 1: chars 0-500
    chunk 2: chars 450-950   ← 50 chars overlap with chunk 1
    chunk 3: chars 900-1400  ← 50 chars overlap with chunk 2
    """
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i: i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── NVIDIA LLAMA CHAT (non-streaming) ────────────────────────────────────────
def ask_nvidia(prompt: str, chat_history: list = []) -> str:
    """
    Two calling modes:

    Mode A — classic (used by router node):
        ask_nvidia("classify this...", [])
        → appends prompt as a user message

    Mode B — messages-only (used by rag_node / chitchat_node):
        ask_nvidia("", full_messages_list)
        → uses chat_history directly (already contains system + history + question)
        → prompt is ignored when it's an empty string
    """
    if prompt:
        messages = chat_history + [{"role": "user", "content": prompt}]
    else:
        messages = chat_history   # agent already built the full message list

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

# ── NVIDIA LLAMA CHAT (streaming) — Step 4 ───────────────────────────────────
def stream_nvidia(messages: list[dict]):
    """
    Generator function that streams the LLM response token by token.

    HOW SSE (Server-Sent Events) WORKS:
    1. Flask sends HTTP response with Content-Type: text/event-stream
    2. This generator yields small text chunks as they arrive from NVIDIA
    3. The browser's JS EventSource reads each chunk and appends it to the UI
    4. User sees text appearing word-by-word — no waiting for full response

    HOW stream=True WORKS with NVIDIA API:
    - Normal call:   NVIDIA waits for full answer → sends one big JSON blob
    - Streaming call: NVIDIA sends partial tokens immediately as SSE lines
      Each line: data: {"choices":[{"delta":{"content":"Hello"}}]}
      Last line:  data: [DONE]

    Yields: plain text token strings (app.py wraps them in SSE format)
    """
    import json

    payload = {
        "model": "meta/llama3-70b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500,
        "stream": True
    }

    with requests.post(
        f"{BASE_URL}/chat/completions",
        headers=HEADERS,
        json=payload,
        stream=True
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            text = line.decode("utf-8")
            if not text.startswith("data:"):
                continue

            data_str = text[len("data:"):].strip()

            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                token = data["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except (json.JSONDecodeError, KeyError, IndexError):
                continue