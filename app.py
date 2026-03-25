print("🚀 App starting...")

# app.py
import os
import time
import re
from datetime import datetime
from flask import Flask, request, render_template
from dotenv import load_dotenv

# Local imports
from utils import load_text, load_pdf, chunk_text, ask_nvidia
from rag_utils import create_embeddings, build_faiss_index, search_faiss
from database import init_db, save_message, load_messages

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./data"

# -------- GLOBAL (FAISS cannot be stored in session) --------
faiss_index = None
chunks_store = []

# -------- INIT DATABASE --------
init_db()

# -------- CLEAN OLD FILES --------
def cleanup_uploads(folder, max_age_seconds=3600):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            os.remove(path)

# -------- EXTRACT SOURCES --------
def extract_sources(text):
    return list(set(re.findall(r'Page\s*\d+', text)))

# -------- AI FUNCTION --------
def ask_question(question):
    global faiss_index, chunks_store

    if not chunks_store or faiss_index is None:
        return "❌ Please upload a document first.", []

    context_chunks = search_faiss(question, chunks_store, faiss_index)

    # Build context text with page numbers
    context_text = ""
    for c in context_chunks:
        context_text += f"{c['chunk']} (Page {c['page']})\n"

    prompt = f"""
    Answer clearly and concisely.
    Quote the exact text from context if available, with page numbers.
    If not found, say: Not available.

    Context:
    {context_text}

    Question:
    {question}
    """

    reply = ask_nvidia(prompt, [])

    # Extract sources
    sources = list(set([f"Page {c['page']}" for c in context_chunks]))

    return reply, sources

# -------- ROUTE --------
@app.route("/", methods=["GET", "POST"])
def index():
    global faiss_index, chunks_store

    cleanup_uploads(app.config['UPLOAD_FOLDER'])

    if request.method == "POST":
        files = request.files.getlist("file")
        question = request.form.get("question", "").strip()

        # -------- FILE UPLOAD --------
        if files and any(f.filename for f in files):
            all_chunks = []

            for file in files:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)

                if path.endswith(".pdf"):
                    text = load_pdf(path)
                else:
                    text = load_text(path)

                chunks = chunk_text(text)
                all_chunks.extend(chunks)

            # 🔥 Create embeddings + FAISS
            embeddings = create_embeddings(all_chunks)
            faiss_index = build_faiss_index(embeddings)

            chunks_store = all_chunks

        # -------- ASK QUESTION --------
        if question:
            current_time = datetime.now().strftime("%H:%M")

            # Save user message
            save_message("user", question, current_time)

            answer, sources = ask_question(question)

            # Save AI response
            save_message("assistant", answer, current_time)

    # Load chat history from DB
    chat_history = load_messages()

    return render_template("index.html", chat_history=chat_history)

# -------- RUN --------
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)