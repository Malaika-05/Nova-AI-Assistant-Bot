from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq
import os
import json
import time
import re
import sqlite3
import uuid
from datetime import datetime

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = "studybot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            subject     TEXT DEFAULT 'General',
            mode        TEXT DEFAULT 'chat',
            created_at  TEXT,
            updated_at  TEXT,
            title       TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            role        TEXT,
            content     TEXT,
            mode        TEXT,
            subject     TEXT,
            timestamp   TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_session(subject, mode, first_message):
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = first_message[:45] + ("..." if len(first_message) > 45 else "")
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (session_id, subject, mode, created_at, updated_at, title) VALUES (?,?,?,?,?,?)",
        (session_id, subject, mode, now, now, title)
    )
    conn.commit()
    conn.close()
    return session_id

def save_message(session_id, role, content, mode, subject):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, mode, subject, timestamp) VALUES (?,?,?,?,?,?)",
        (session_id, role, content, mode, subject, now)
    )
    conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))
    conn.commit()
    conn.close()

# ── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPTS = {
    "chat": """You are Nova, a super friendly and clever AI study buddy for university students.

Your vibe:
- Talk like a smart friend, not a boring textbook
- Use simple words. If a concept is hard, break it down with a fun analogy
- Add encouragement naturally
- Use emojis occasionally 🎯
- Never make the student feel dumb

How you structure answers:
- Start with a one-line simple answer (the short version)
- Then explain in detail with a real-life example
- End with a quick tip
- Use **bold** for key terms, bullet points for lists""",

    "quiz": """You are Nova, a fun quiz master. Generate exactly 5 multiple choice questions.

IMPORTANT: Follow this EXACT format for every question:

Q1: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter A/B/C/D]
EXPLANATION: [one sentence]
###
Q2: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter]
EXPLANATION: [one sentence]
###
Q3: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter]
EXPLANATION: [one sentence]
###
Q4: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter]
EXPLANATION: [one sentence]
###
Q5: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter]
EXPLANATION: [one sentence]
###

Rules: ANSWER must be single letter only. No text before Q1 or after last ###""",

    "summarize": """You are Nova, a study buddy who makes revision easy and fun.

When summarizing:
🎯 **One-line definition** — explain like the student is 15
📌 **Key points** — 5 to 7 bullet points, no jargon
🌍 **Real-world example** — something relatable
⚠️ **Common mistake** — one thing students get wrong
💡 **Remember this** — one catchy sentence

Keep it friendly, clear, and exam-ready.""",

    "solve": """You are Nova, a patient math and CS tutor.

When solving:
- Say what type of problem it is
- List what information is given
- Solve step by step with numbered steps
- After each step, add a plain English explanation in brackets
- Show final answer clearly with ✅
- End with: "Does that make sense? Let me know if any step is confusing!"

Never skip steps.""",

    "flashcard": """You are Nova. Generate exactly 5 flashcards on the given topic.

Use EXACTLY this format:

CARD1
FRONT: [short question or term, max 8 words]
BACK: [clear answer, max 20 words]
CARD2
FRONT: [short question or term, max 8 words]
BACK: [clear answer, max 20 words]
CARD3
FRONT: [short question or term, max 8 words]
BACK: [clear answer, max 20 words]
CARD4
FRONT: [short question or term, max 8 words]
BACK: [clear answer, max 20 words]
CARD5
FRONT: [short question or term, max 8 words]
BACK: [clear answer, max 20 words]

Rules: No extra text. FRONT and BACK on separate lines. Keep text SHORT."""
}

# ── STATE ─────────────────────────────────────────────────────────────────────
conversation_history = []
current_subject = "General"
current_mode = "chat"
current_session_id = None

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global current_subject, current_mode, conversation_history, current_session_id
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        current_mode = data.get("mode", "chat")
        current_subject = data.get("subject", "General")

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Create session on first message
        if current_session_id is None:
            current_session_id = create_session(current_subject, current_mode, user_message)

        system = PROMPTS.get(current_mode, PROMPTS["chat"])
        if current_subject != "General":
            system += f"\n\nStudent is studying: {current_subject}. Keep examples relevant."

        conversation_history.append({"role": "user", "content": user_message})

        # Keep last 6 messages to save tokens
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]

        # API call with rate-limit retry
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": system}] + conversation_history,
                    max_tokens=1000,
                    temperature=0.7
                )
                break
            except Exception as api_err:
                err_str = str(api_err)
                if "429" in err_str and attempt < 2:
                    match = re.search(r'try again in (\d+\.?\d*)s', err_str)
                    wait = float(match.group(1)) + 0.5 if match else 4.0
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    conversation_history.pop()
                    if "429" in err_str:
                        return jsonify({"reply": "⏳ Too many requests. Wait a moment and try again!"}), 200
                    raise

        if not response:
            conversation_history.pop()
            return jsonify({"reply": "⏳ No response received. Please try again."}), 200

        reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": reply})

        # Save to DB
        save_message(current_session_id, "user",      user_message, current_mode, current_subject)
        save_message(current_session_id, "assistant", reply,        current_mode, current_subject)

        return jsonify({"reply": reply, "mode": current_mode, "session_id": current_session_id})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"reply": f"Oops! Something went wrong: {str(e)}"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, current_session_id
    conversation_history = []
    current_session_id = None
    return jsonify({"status": "reset"})


@app.route("/sessions", methods=["GET"])
def get_sessions():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT session_id, title, subject, mode, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 30"
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        print("Sessions error:", e)
        return jsonify([]), 200


@app.route("/sessions/<session_id>", methods=["GET"])
def get_session_messages(session_id):
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT role, content, mode, subject, timestamp FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        print("Session messages error:", e)
        return jsonify([]), 200


@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        conn = get_db()
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "deleted"})
    except Exception as e:
        print("Delete error:", e)
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
