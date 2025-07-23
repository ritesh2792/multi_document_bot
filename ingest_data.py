import os
from glob import glob
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma  # ← alt. if FAISS gives trouble

# --------------------
# 0. Setup
# --------------------
load_dotenv(override=True)
EMBED_MODEL = "text-embedding-3-small"  # cheaper & better than ada-002
DATA_DIR = "./"                         # where PDFs live
PARQUET_PATH = "pdf_chunks.parquet"
CHROMA_DIR = "chroma_store"

# --------------------
# 1. Collect PDFs
# --------------------
pdf_paths = glob(os.path.join(DATA_DIR, "*.pdf"))
if not pdf_paths:
    raise SystemExit("No PDFs found in current directory.")

# --------------------
# 2. Splitter & embedder
# --------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
embedder = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# 3. Load, split, embed
# --------------------
records = []
all_texts = []
all_metadatas = []

for pdf_file in pdf_paths:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        uid = str(uuid4())
        content = chunk.page_content
        meta = {
            "id": uid,
            "pdf_name": os.path.basename(pdf_file),
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page", None),
        }
        records.append({"id": uid, "pdf_name": meta["pdf_name"], "content": content})
        all_texts.append(content)
        all_metadatas.append(meta)

# --------------------
# 4. Persist tabular copy (Parquet instead of Delta)
# --------------------
df = pd.DataFrame(records)
df.to_parquet(PARQUET_PATH, index=False)
print(f"Saved {len(df)} chunks to {PARQUET_PATH}")

# --------------------
# 5. Build and persist Chromas vector store
# --------------------
vectorstore = Chroma.from_texts(
    texts=all_texts,
    embedding=embedder,
    metadatas=all_metadatas,
    persist_directory=CHROMA_DIR
)
vectorstore.persist()
print(f"Chroma DB saved to {CHROMA_DIR}")
