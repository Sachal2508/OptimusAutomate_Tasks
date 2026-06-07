"""
Task 2: Chatbot with Intent Recognition — Streamlit UI
CodeAlpha AI Internship
Run with: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chatbot_engine import ChatbotEngine

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="UniBot — University FAQ Chatbot",
    page_icon="🎓",
    layout="centered",
)

# ── Minimal CSS (no HTML chat rendering) ────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0d1117; }
.stChatMessage { background-color: #161b22 !important; border: 1px solid #30363d; border-radius: 10px; }
.stTextInput > div > div > input {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 8px;
}
.stButton > button {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 8px;
}
.stButton > button:hover {
    border-color: #58a6ff;
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = ChatbotEngine()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm **UniBot**, your university FAQ assistant. Ask me about admissions, fees, programs, scholarships, exams, hostels, and more!",
            "intent": None,
            "confidence": None,
        }
    ]

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 UniBot")

    st.markdown("### 📌 What I can help with")
    topics = ["Admissions & Eligibility", "Fee Structure", "Programs & Departments",
              "Scholarships & Financial Aid", "GPA & Grading Policy", "Course Registration",
              "Hostel & Accommodation", "Library Resources", "Exam Schedule",
              "Career Services", "Mental Health Support"]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.markdown("Uses **TF-IDF vectorization** + **Cosine Similarity** for intent recognition. Tracks multi-turn context so follow-ups like *'tell me more'* work naturally.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.engine.reset()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! How can I help you?",
                "intent": None,
                "confidence": None,
            }
        ]
        st.rerun()

    st.markdown(f"💬 **Messages:** {len(st.session_state.messages)}")

# ── Title ────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#58a6ff;'>🎓 UniBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#8b949e;'>University FAQ Chatbot · Intent Recognition · Multi-turn Conversation</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Render chat history using st.chat_message ────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("intent") and msg["intent"] not in (None, "greeting"):
            st.caption(f"Intent: `{msg['intent']}` · Confidence: `{msg['confidence']:.0%}`")

# ── Quick-reply chips ─────────────────────────────────────────────
st.markdown("**Quick questions:**")
chips = ["Admissions", "Fee Structure", "Scholarships", "Programs", "Exam Schedule", "Hostel", "Contact"]
cols = st.columns(len(chips))
chip_clicked = None
for col, chip in zip(cols, chips):
    with col:
        if st.button(chip, key=f"chip_{chip}", use_container_width=True):
            chip_clicked = chip

# ── Chat input ────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about university...")

# ── Handle message ────────────────────────────────────────────────
def handle_message(text: str):
    text = text.strip()
    if not text:
        return
    # Add user message
    st.session_state.messages.append({"role": "user", "content": text, "intent": None, "confidence": None})
    # Get bot response
    result = st.session_state.engine.chat(text)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "intent": result["intent"],
        "confidence": result["confidence"],
    })
    st.rerun()

if user_input:
    handle_message(user_input)
elif chip_clicked:
    handle_message(chip_clicked)