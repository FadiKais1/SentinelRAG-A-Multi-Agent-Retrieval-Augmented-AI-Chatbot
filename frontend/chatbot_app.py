# import sys
# import os

# # Add project root to PYTHONPATH
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(ROOT_DIR)

# import streamlit as st
# from backend.rag_pipeline import index_all_documents, answer_query

# st.set_page_config(page_title="Multi-Agent RAG Chatbot", layout="wide")

# st.title("🤖 Multi-Agent RAG Chatbot")
# st.write("Ask questions about your documents. The system uses Retrieval + Multi-Agent AI Workflow.")

# # Sidebar for indexing
# with st.sidebar:
#     st.header("📄 Document Indexing")

#     if st.button("Reindex Documents"):
#         st.write("Indexing...")
#         count = index_all_documents()
#         st.success(f"Indexed {count} chunks!")

# # Chat interface
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.text_input("Ask something about your documents:")

# if st.button("Send") and user_input.strip():
#     with st.spinner("Thinking..."):
#         result = answer_query(user_input)

#         st.session_state.chat_history.append({
#             "question": user_input,
#             "answer": result["answer"],
#             "critique": result.get("critique", ""),
#             "sources": result.get("context", [])
#         })

# # Display chat
# for msg in st.session_state.chat_history[::-1]:
#     st.markdown(f"### ❓ Question\n{msg['question']}")
#     st.markdown(f"### 🧠 Answer\n{msg['answer']}")
#     st.markdown(f"### 🔍 Critique\n{msg['critique']}")

#     st.markdown("### 📄 Sources")
#     for s in msg["sources"]:
#         st.markdown(f"- **{s['metadata']['source']}** (chunk {s['metadata']['chunk_index']})")

#     st.markdown("---")


import sys
import os

# Add project root to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
print("PYTHON EXECUTABLE:", sys.executable)

import streamlit as st
from backend.rag_pipeline import index_all_documents, answer_query


DOCS_DIR = os.path.join(ROOT_DIR, "data", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="Multi-Agent RAG Chatbot", layout="wide")

st.title("🤖 Multi-Agent RAG Chatbot")
st.write("Ask questions about your documents. The system uses Retrieval + Multi-Agent AI workflow with conversation memory.")


# ----------------------------
# Sidebar: document tools
# ----------------------------
with st.sidebar:
    st.header("📄 Document Tools")

    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(DOCS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success("Files uploaded. Click 'Reindex Documents' to include them.")

    if st.button("Reindex Documents"):
        with st.spinner("Indexing documents..."):
            count = index_all_documents()
        st.success(f"Indexed {count} chunks.")


# ----------------------------
# Session state for chat
# ----------------------------
if "chat_history" not in st.session_state:
    # For display: list of dicts with question/answer/critique/sources
    st.session_state.chat_history = []

if "llm_history" not in st.session_state:
    # For LLM memory: list of {"user": "...", "assistant": "..."}
    st.session_state.llm_history = []


# ----------------------------
# Chat input
# ----------------------------
user_input = st.text_input("Ask something about your documents (or anything):", key="user_input")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        result = answer_query(user_input, history=st.session_state.llm_history)

        # Update LLM memory
        st.session_state.llm_history = result.get("history", st.session_state.llm_history)

        st.session_state.chat_history.append(
            {
                "question": user_input,
                "answer": result["answer"],
                "critique": result.get("critique", ""),
                "sources": result.get("context", []),
                "mode": result.get("mode", "unknown"),
            }
        )

# ----------------------------
# Display conversation
# ----------------------------
for msg in reversed(st.session_state.chat_history):
    st.markdown(f"### ❓ Question")
    st.markdown(msg["question"])

    st.markdown(f"### 🧠 Answer")
    st.markdown(msg["answer"])

    if msg["critique"]:
        st.markdown(f"### 🔍 Critique")
        st.markdown(msg["critique"])

    st.markdown(f"**Mode:** `{msg['mode']}`")

    if msg["sources"]:
        st.markdown("### 📄 Sources")
        for s in msg["sources"]:
            src = s["metadata"].get("source", "unknown")
            idx = s["metadata"].get("chunk_index", "?")
            st.markdown(f"- **{src}** (chunk {idx})")

    st.markdown("---")
