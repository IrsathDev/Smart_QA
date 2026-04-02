# app.py
# Step 4 Upgrade: Streaming SSE + Pydantic Validation + LangSmith Tracing
# Bug fix: PRG (Post-Redirect-Get) pattern — prevents duplicate submissions on reload

print("🚀 App starting...")

import os
import re
import time
import json
from datetime import datetime
from flask import Flask, request, render_template, Response, stream_with_context, redirect, url_for, session
from dotenv import load_dotenv

from utils import load_text, load_pdf, chunk_text, stream_nvidia
from rag_utils import build_hybrid_index, hybrid_search
from agent import run_agent
from database import init_db, save_message, load_messages, load_history_for_llm
from schemas import RAGResponse, UploadResponse

load_dotenv()

# ── LANGSMITH TRACING SETUP ───────────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "smart-qa-app"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
app.config['UPLOAD_FOLDER'] = "./data"

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
chroma_collection = None
bm25_index        = None
chunks_store      = []

init_db()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def cleanup_uploads(folder, max_age_seconds=3600):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            os.remove(path)

def extract_sources(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'Page\s*\d+', text)))


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING ROUTE
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/stream")
def stream():
    global chroma_collection, bm25_index, chunks_store

    question = request.args.get("question", "")
    route    = request.args.get("route", "rag")

    if not question:
        return Response("data: [DONE]\n\n", mimetype="text/event-stream")

    history = load_history_for_llm(last_n=10)

    if route == "rag" and chroma_collection and bm25_index and chunks_store:
        chunks = hybrid_search(
            query=question,
            all_chunks=chunks_store,
            chroma_collection=chroma_collection,
            bm25_index=bm25_index,
            top_k=5
        )
        context = "\n\n".join(chunks)
        system_content = (
            "You are a helpful assistant. Answer using ONLY the document context.\n"
            "Mention page numbers like (Page 2) if available.\n"
            "If not found, say: 'Not available in the document.'\n\n"
            f"Context:\n{context}"
        )
    else:
        system_content = (
            "You are a friendly, helpful assistant. "
            "Answer conversationally. Keep it short and warm."
        )

    messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": question}]
    )

    def generate():
        full_answer = []

        for token in stream_nvidia(messages):
            full_answer.append(token)
            yield f"data: {json.dumps(token)}\n\n"

        yield "data: [DONE]\n\n"

        complete = "".join(full_answer).replace("[Page", "📄 Page")
        sources  = extract_sources(complete)

        try:
            validated = RAGResponse(answer=complete, sources=sources, route=route)
            save_message("assistant", validated.answer, datetime.now().strftime("%H:%M"))
            print(f"✅ Validated — route={validated.route}, sources={validated.sources}")
        except Exception as e:
            print(f"⚠️  Pydantic validation failed: {e}")
            save_message("assistant", complete, datetime.now().strftime("%H:%M"))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTE — PRG pattern (Post → Redirect → Get)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def index():
    global chroma_collection, bm25_index, chunks_store

    # ── GET: render page, pick up any flash data from session ─────────────────
    if request.method == "GET":
        upload_status   = session.pop("upload_status", None)
        stream_question = session.pop("stream_question", None)
        agent_route     = session.pop("agent_route", None)

        cleanup_uploads(app.config['UPLOAD_FOLDER'])
        chat_history = load_messages()

        return render_template(
            "index.html",
            chat_history=chat_history,
            upload_status=upload_status,
            agent_route=agent_route,
            stream_question=stream_question   # JS reads this ONCE then page is clean
        )

    # ── POST: process, store results in session, REDIRECT to GET ──────────────
    # This is the PRG pattern — after POST we always redirect to GET.
    # This means browser reload will do a GET, not re-submit the POST.
    files    = request.files.getlist("file")
    question = request.form.get("question", "").strip()

    # ── File upload ───────────────────────────────────────────────────────────
    if files and any(f.filename for f in files):
        all_chunks = []
        for file in files:
            if not file.filename:
                continue
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            text   = load_pdf(path) if path.lower().endswith(".pdf") else load_text(path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        if all_chunks:
            chroma_collection, bm25_index = build_hybrid_index(all_chunks)
            chunks_store = all_chunks
            try:
                upload_resp = UploadResponse(
                    success=True,
                    chunk_count=len(all_chunks),
                    message=f"Indexed {len(all_chunks)} chunks — hybrid + streaming ready."
                )
                session["upload_status"] = f"✅ {upload_resp.message}"
            except Exception as e:
                session["upload_status"] = f"⚠️ Upload issue: {e}"

    # ── Question ──────────────────────────────────────────────────────────────
    if question:
        save_message("user", question, datetime.now().strftime("%H:%M"))

        _, agent_route = run_agent(
            question=question,
            history=load_history_for_llm(last_n=10),
            chroma_collection=chroma_collection,
            bm25_index=bm25_index,
            chunks_store=chunks_store
        )

        # Store in session — the GET will read these ONCE and clear them
        session["stream_question"] = question
        session["agent_route"]     = agent_route

    # ── Always redirect to GET after POST ────────────────────────────────────
    return redirect(url_for("index"))


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True,
        threaded=True
    )