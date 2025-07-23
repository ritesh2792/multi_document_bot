# app.py
import streamlit as st
import tempfile
from get_response import get_response_chat, add_pdf_to_store

st.set_page_config(page_title="Retail Policy Assist", layout="wide")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚙️ Settings")
    # --- New: Upload PDFs ---
    uploaded = st.file_uploader("Upload any PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Embed & Add to KB"):
            with st.spinner("Embedding and updating index..."):
                for uf in uploaded:
                    # save to a temp file so PyPDFLoader can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    add_pdf_to_store(tmp_path)
            st.success("PDF(s) ingested successfully!")

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        #st.experimental_rerun()

st.title("🛍️ Retail Policy Assist")

# Render chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
user_query = st.chat_input("Ask about retail/vendor policies...")

if user_query:
    # user bubble
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # bot answer
    with st.spinner("Thinking..."):
        resp = get_response_chat(user_query)
    answer = resp["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
