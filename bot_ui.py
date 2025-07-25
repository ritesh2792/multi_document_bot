import os
import streamlit as st
import tempfile
import pandas as pd

from get_response import get_response_chat, add_pdf_to_store
from agents.content_agents import (
    parse_request,
    social_media_agent,
    website_listing_agent,
    video_ad_agent
)

st.set_page_config(page_title="Retail + Content Bot", layout="wide")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings & Data")
    bot_type = st.radio("Select Bot", ["Knowledge Bot", "Content Generator"])

    st.markdown("---")
    # ← Add your “Clear conversation” button here:
    if st.button("🧹 Clear conversation"):
        if bot_type == "Knowledge Bot":
            st.session_state.hist_knowledge = []
        else:
            st.session_state.hist_content = []
            
    st.markdown("---")
    if bot_type == "Knowledge Bot":
        st.subheader("Knowledge KB")
        pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if pdfs and st.button("Embed & Add to KB"):
            with st.spinner("Embedding PDFs..."):
                for p in pdfs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(p.read())
                        add_pdf_to_store(tmp.name)
            st.success("PDF(s) ingested!")
    
    else:
        st.subheader("Product Data")
        excel = st.file_uploader("Upload Products Excel", type=["xlsx"])
        if excel and st.button("Load Products"):
            os.makedirs("data", exist_ok=True)
            # write raw bytes
            with open("data/products.xlsx", "wb") as f:
                f.write(excel.getbuffer())
            st.success("products.xlsx saved to ./data/")
    
    
        

# ─── Main Pane: Chat UI ───────────────────────────────────────────────────────
st.title("💬 Chat")

# Initialize histories
st.session_state.setdefault("hist_knowledge", [])
st.session_state.setdefault("hist_content", [])

if bot_type == "Knowledge Bot":
    history = st.session_state.hist_knowledge
    # Render chat history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about retail/vendor policies…")
    if user_q:
        history.append({"role": "user", "content": user_q})
        st.chat_message("user").markdown(user_q)

        with st.spinner("Thinking…"):
            resp = get_response_chat(user_q)
        answer = resp["answer"]

        history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

else:
    history = st.session_state.hist_content
    # Render chat history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("E.g. “Social Media post for P012” or “Video Ad for Red Apple”…")
    if user_q:
        history.append({"role": "user", "content": user_q})
        st.chat_message("user").markdown(user_q)

        # 1) Parse out ID/Name + content_type
        parsed = parse_request(user_q)
        item_id = parsed.get("item_id")
        item_name = parsed.get("item_name")
        ctype   = parsed.get("content_type")

        # 2) Load products & lookup
        data_path = "data/products.xlsx"
        if not os.path.exists(data_path):
            answer = "❌ No products.xlsx found in ./data. Please upload in sidebar."
        elif not (item_id or item_name) or ctype not in ("Social Media","Website Listing","Video Ad"):
            answer = ("❌ Could not find both a valid item_id/name and/or content_type in your request. "
                      "Please phrase like “Social Media post for P012” or “Website Listing for Red Apple.”")
        else:
            df = pd.read_excel(data_path)
            if item_id:
                df_filtered = df[df["item_id"].astype(str).str.lower() == item_id.lower()]
            else:
                df_filtered = df[df["item_name"].str.lower() == item_name.lower()]

            if df_filtered.empty:
                answer = "❌ No matching product found."
            else:
                row = df_filtered.iloc[0]
                product = {
                    "name":     row["item_name"],
                    "features": row.get("description", ""),
                    "specs":    f"Color: {row['color']}, Size: {row['size']}",
                    "usp":      f"Pack: {row['pack_item']}, Ships via {row['shipping_mode']}",
                    "audience": "General",
                    "price":    row["selling_price"],
                }
                # 3) Generate
                if ctype == "Social Media":
                    answer = social_media_agent(product)
                elif ctype == "Website Listing":
                    answer = website_listing_agent(product)
                else:
                    answer = video_ad_agent(product)

        history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
