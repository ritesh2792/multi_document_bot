Retail Policy Assist

Tagline: From page to answer in a single prompt.

A lightweight Retrieval‑Augmented Generation (RAG) chatbot that lets retail teams ask natural‑language questions over vendor contracts and policy PDFs. Built with Python, Streamlit, LangChain, OpenAI, and ChromaDB—no Databricks or Spark required.

What This POC Does?

Uploads & ingests PDFs from the UI or file system
Chunks & embeds contract text using OpenAI embeddings
Stores vectors locally in Chroma (persisted on disk)
Retrieves relevant snippets and feeds them to an LLM
Answers conversationally in a chat-style Streamlit UI with full history

Architecture at a Glance

Vendor PDFs  ──► Split & Embed (LangChain/OpenAI) ──► Chroma Vector DB
                                                ▲
User ── Streamlit Chat UI ──► get_response.py ──┘─► LLM (OpenAI GPT-*) ──► Answer. 

Use the provided diagram PNGs/SVGs in your deck.)

🗂 Project Structureproject/
├─ app.py                 # Streamlit chat UI
├─ get_response.py        # Retrieval + LLM answering; add_pdf_to_store()
├─ ingest_data.py         # (Optional) standalone batch ingestion script
├─ build_embeddings.py    # One-time bulk build (if you prefer CLI)
├─ chroma_store/          # Chroma persistent DB (auto-created)
├─ pdf_chunks.parquet     # Tabular metadata of chunks
├─ .env                   # OPENAI_API_KEY, etc.
└─ requirements.txt🔧 PrerequisitesPython 3.9–3.11

An OpenAI API key (OPENAI_API_KEY)
No native libs needed if you stick with Chroma (pure Python install)
Install deps

pip install -r requirements.txt

Create a .env

OPENAI_API_KEY=sk-your-key 

Ingest PDFs (Two Ways)

1. Through the UI (recommended for demo)Open the app (see Run the app below)
Sidebar → Upload vendor PDFs → Embed & Add to KB
Start chatting; new docs are immediately searchable

2. CLI script (bulk/offline)python build_embeddings.pyThis locates all *.pdf in the current folder, embeds, and persists to chroma_store/ & pdf_chunks.parquet.

💬 Run the Appstreamlit run app.pyOpen the URL shown (usually http://localhost:8501). 

Ask a question like:
“What are the payment terms for Vendor A?”
🧩 Key Modules

get_response.pyLoads Chroma + embedder once
get_response_chat(query: str) -> {"answer": str}: retrieves context and calls the LLM
add_pdf_to_store(path: str): helper used by the UI to ingest new PDFs on the fly
app.py -->Uses st.chat_message/st.chat_input for a true chat UX
Maintains conversation history in st.session_state
Sidebar controls for upload & clearing the convo

Security & GovernanceSecrets via .env (no keys in code)
Local vector DB avoids third-party lock-in; can run behind your firewall
Prompt enforces “answer only from context” to reduce hallucinations
🛠 TroubleshootingIssueLikely CauseFix"ModuleNotFoundError: langchain_openai"Version mismatchpip install langchain-openai==0.1.7Chroma dir missingFirst run not ingestedUpload a PDF or run build_embeddings.pyBad answers / hallucinationMissing contextIncrease TOP_K, improve chunk size, or re‑embedOpenAI rate limitRapid queriesAdd retry/backoff, cache embeddings🧭 Roadmap IdeasSnippet highlighting & citations in answers
Role-based access, audit logging
Multi-source ingest (SharePoint, DB tables, email)
Export answers/snippets to PDF or Excel
Prompt templates per department (Legal, Procurement, Finance)
📄 License / UsageInternal POC only. Add your company’s standard notice or OSS license if you open source.