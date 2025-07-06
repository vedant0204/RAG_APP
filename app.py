import streamlit as st
from rag_logic import process_and_store_doc, answer_question
import os
import nltk
nltk.download('punkt')

st.set_page_config(page_title="Sir Reads-a-Lot", layout="wide")

st.markdown("""
    <style>
        /* Page background and font */
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }

        /* Centered Title */
        .main-title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #f0f0f0;
            margin-top: 20px;
        }

        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #cccccc;
            margin-bottom: 30px;
        }

        /* Hide default labels */
        .css-9s5bis, .css-1y4p8pa {
            display: none;
        }

        /* Chat history styling */
        .chat-message {
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 10px;
            padding-left: 15px;
        }

        .user-message {
            color: #ffffff;
        }

        .bot-message {
            color: #dcdcdc;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Upload
st.sidebar.title("Upload Your Document")
st.sidebar.markdown("Supported formats: .txt, .pdf, .md")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed"
)

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    process_and_store_doc(file_path)
    st.sidebar.success("Document uploaded and processed successfully.")

# Title and Subtitle
st.markdown('<div class="main-title">Sir Reads-a-Lot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent assistant for document-based queries</div>', unsafe_allow_html=True)

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
query = st.text_input("How can I assist you today?", key="query_input")

# Handle user query
if query:
    st.session_state.messages.append(("user", query))
    response, sources = answer_question(query)
    st.session_state.messages.append(("bot", response))

# Chat history display
for sender, msg in st.session_state.messages:
    if sender == "user":
        st.markdown(f'<div class="chat-message user-message">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">{msg}</div>', unsafe_allow_html=True)
    