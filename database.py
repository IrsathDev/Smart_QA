import sqlite3

def init_db():
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_message(role, content, time):
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, time)
    )

    conn.commit()
    conn.close()

def load_messages():
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()

    cursor.execute("SELECT role, content, timestamp FROM chats")
    rows = cursor.fetchall()

    conn.close()

    return [{"role": r[0], "content": r[1], "time": r[2]} for r in rows]