# database.py
# Step 3 Upgrade: Added load_history_for_llm() for agent memory

import sqlite3


def init_db():
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT,
            content   TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_message(role: str, content: str, time: str):
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, time)
    )
    conn.commit()
    conn.close()


def load_messages() -> list[dict]:
    """Load all messages for the chat UI (role, content, time)."""
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp FROM chats ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "time": r[2]} for r in rows]


# ── NEW in Step 3 ─────────────────────────────────────────────────────────────
def load_history_for_llm(last_n: int = 10) -> list[dict]:
    """
    Loads the last N chat turns formatted for the LLM messages API.
    This is what gets injected into the agent as conversation memory.

    Why last 10? LLMs have a context window limit. Keeping only recent
    turns prevents the prompt from growing too large while still giving
    the agent enough context to answer follow-up questions correctly.

    Returns list of dicts like: [{"role": "user", "content": "..."}, ...]
    """
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chats ORDER BY id DESC LIMIT ?",
        (last_n,)
    )
    rows = cursor.fetchall()
    conn.close()

    # Reverse so oldest turn comes first (correct chronological order for LLM)
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]