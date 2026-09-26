import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import os
import time
import re
import sqlite3
import uuid
from datetime import datetime

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nova - AI Study Buddy", page_icon="🎯", layout="wide")

client = Groq(api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", ""))

DB_PATH = "studybot.db"
MODEL_NAME = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated by Groq

# ── DATABASE ───────────────────────────────────────────────────────────────
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


def get_sessions():
    conn = get_db()
    rows = conn.execute(
        "SELECT session_id, title, subject, mode, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_messages(session_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, mode, subject, timestamp FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_session(session_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


init_db()

# ── PROMPTS ────────────────────────────────────────────────────────────────
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

MODES = list(PROMPTS.keys())

# ── SESSION STATE ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role": "user"/"assistant", "content": ...}]
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "subject" not in st.session_state:
    st.session_state.subject = "General"


def start_new_chat():
    st.session_state.messages = []
    st.session_state.current_session_id = None


def load_chat(session_id, mode, subject):
    st.session_state.current_session_id = session_id
    st.session_state.mode = mode
    st.session_state.subject = subject
    rows = get_session_messages(session_id)
    st.session_state.messages = [{"role": r["role"], "content": r["content"]} for r in rows]


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Nova")
    st.caption("Your AI study buddy")

    if st.button("➕ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.session_state.mode = st.selectbox(
        "Mode", MODES, index=MODES.index(st.session_state.mode),
        format_func=lambda m: m.capitalize()
    )
    st.session_state.subject = st.text_input("Subject", value=st.session_state.subject)

    st.divider()
    st.caption("Recent sessions")

    for s in get_sessions():
        col1, col2 = st.columns([5, 1])
        label = f"{s['title'] or '(untitled)'}"
        active = s["session_id"] == st.session_state.current_session_id
        with col1:
            if st.button(("👉 " if active else "") + label, key=f"load_{s['session_id']}", use_container_width=True):
                load_chat(s["session_id"], s["mode"], s["subject"])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s['session_id']}"):
                delete_session(s["session_id"])
                if st.session_state.current_session_id == s["session_id"]:
                    start_new_chat()
                st.rerun()

# ── MAIN CHAT AREA ─────────────────────────────────────────────────────────
st.header("Nova — AI Study Buddy 🎯")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_message = st.chat_input("Ask Nova anything...")

if user_message:
    mode = st.session_state.mode
    subject = st.session_state.subject

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_session(subject, mode, user_message)

    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    system = PROMPTS.get(mode, PROMPTS["chat"])
    if subject != "General":
        system += f"\n\nStudent is studying: {subject}. Keep examples relevant."

    # keep last 6 messages to save tokens
    api_history = st.session_state.messages[-6:]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        reply = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": system}] + api_history,
                    max_tokens=1000,
                    temperature=0.7
                )
                reply = response.choices[0].message.content
                break
            except Exception as api_err:
                err_str = str(api_err)
                if "429" in err_str and attempt < 2:
                    match = re.search(r'try again in (\d+\.?\d*)s', err_str)
                    wait = float(match.group(1)) + 0.5 if match else 4.0
                    placeholder.markdown(f"⏳ Rate limited, retrying in {wait:.0f}s...")
                    time.sleep(wait)
                elif "429" in err_str:
                    reply = "⏳ Too many requests. Wait a moment and try again!"
                else:
                    reply = f"Oops! Something went wrong: {err_str}"

        if reply is None:
            reply = "⏳ No response received. Please try again."

        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    save_message(st.session_state.current_session_id, "user", user_message, mode, subject)
    save_message(st.session_state.current_session_id, "assistant", reply, mode, subject)