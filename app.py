print("🚀 App starting...")

# app.py
import os
import json
import time
from flask import Flask, request, render_template, session
from flask_session import Session
from dotenv import load_dotenv
from utils import load_text, load_pdf, chunk_text, get_relevant_chunks, ask_nvidia

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./data"
app.config['SECRET_KEY'] = "supersecretkey"
app.config['SESSION_TYPE'] = "filesystem"
Session(app)

# Clean old files
def cleanup_uploads(folder, max_age_seconds=3600):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            os.remove(path)

# Handle question with keyword-based chunk selection
def ask_question(question):
    try:
        chat_history = session.get("chat_history", [])
        chunks = session.get("chunks", [])

        if not chunks:
            return "❌ Please upload a document first."

        context_chunks = get_relevant_chunks(question, chunks)
        context = " ".join(context_chunks)

        prompt = f"""
        Return answer in JSON format:
        {{
          "answer": "...",
          "confidence": "high/medium/low"
        }}
        Context:
        {context}
        Question:
        {question}
        """

        reply = ask_nvidia(prompt, chat_history)

        # Save chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": reply})
        session["chat_history"] = chat_history

        try:
            parsed = json.loads(reply)
            return f"{parsed['answer']} (Confidence: {parsed['confidence']})"
        except:
            return reply

    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    uploaded_files = []
    cleanup_uploads(app.config['UPLOAD_FOLDER'])

    if request.method == "POST":
        files = request.files.getlist("file")
        question = request.form.get("question", "")
        all_chunks = []

        if files and any(f.filename for f in files):
            for file in files:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                uploaded_files.append(file.filename)

                # Load content
                if filename.endswith(".pdf"):
                    text = load_pdf(filename)
                else:
                    text = load_text(filename)

                chunks = chunk_text(text)
                all_chunks.extend(chunks)

            # Save chunks & reset chat memory
            session["chunks"] = all_chunks
            session["chat_history"] = []

        if question:
            answer = ask_question(question)

    return render_template("index.html", answer=answer, files=uploaded_files)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)