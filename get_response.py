# get_response.py
import os
import json
from uuid import uuid4
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage
#--- add to get_response.py ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma  # already used

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
# -----------------------------
# 0. Config & setup
# -----------------------------
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"          # better/cheaper than ada-002
ANSWER_MODEL = "gpt-4o-mini"                    # or gpt-3.5-turbo if you prefer
RERANK_MODEL = "gpt-3.5-turbo"
EVAL_MODEL   = "gpt-3.5-turbo"

CHROMA_DIR = "chroma_store"                   # created by your build script
PARQUET_PATH = "pdf_chunks.parquet"             # created by your build script

# -----------------------------
# 1. Load vector store & table
# -----------------------------
def _load_store():
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    store = Chroma(
            embedding_function=embedder,
            persist_directory=CHROMA_DIR
        )
    df = pd.read_parquet(PARQUET_PATH)
    return store, df, embedder

    

STORE, DF, EMBEDDER = _load_store()

def add_pdf_to_store(file_path: str):
    """Load, chunk, embed, and persist a new PDF into the existing Chroma DB."""
    docs = PyPDFLoader(file_path).load()
    chunks = SPLITTER.split_documents(docs)
    texts = [c.page_content for c in chunks]
    metas = [{"source": file_path, **c.metadata} for c in chunks]
    STORE.add_texts(texts=texts, metadatas=metas)
    STORE.persist()

# -----------------------------
# 2. Rerank function (unchanged idea)
# -----------------------------
def rerank_rows(query_text: str, rows: List[List[Any]], top_k: int = 10) -> List[List[Any]]:
    """
    rows -> [[id, pdf_name, content, score], ...]
    Uses LLM to rank snippet relevance (returns top_k).
    """
    snippets_list = "\n".join(f"{i+1}. {row[2]}" for i, row in enumerate(rows))
    rerank_prompt = f"""
Re-rank the following document snippets by relevance to the query.
Query: {query_text}

Snippets:
{snippets_list}

Return ONLY a JSON array of the snippet numbers, highest relevance first. Example: [3,1,2]
"""

    llm_ranker = ChatOpenAI(model_name=RERANK_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    resp = llm_ranker([HumanMessage(content=rerank_prompt)]).content.strip()

    try:
        order = json.loads(resp)
    except json.JSONDecodeError:
        order = list(range(1, len(rows) + 1))  # fallback

    reordered = []
    for idx in order:
        if 1 <= idx <= len(rows):
            reordered.append(rows[idx - 1])

    return reordered[:top_k]


# -----------------------------
# 3. Evaluation function
# -----------------------------
def evaluate_response(query_text: str, context: str, answer: str) -> Dict[str, Any]:
    eval_prompt = f"""
Evaluate the assistant's answer for the following query.

Query:
{query_text}

Context used by the assistant:
{context}

Assistant's Answer:
{answer}

Rate the answer on a scale of 1 (poor) to 5 (excellent) and provide a brief justification.
Return ONLY a JSON object with keys "rating" (int) and "comments" (string).
"""
    llm_eval = ChatOpenAI(model_name=EVAL_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    resp = llm_eval([HumanMessage(content=eval_prompt)]).content.strip()
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return {"rating": None, "comments": resp}


# -----------------------------
# 4. Retrieval + context builder
# -----------------------------
def get_context(query_text: str) -> Tuple[str, str]:
    """
    Retrieve top hits from FAISS, rerank, then produce:
     - combined_context (string)
     - results_json (string)
    """
    # similarity_search_with_score returns (Document, score)
    # When retrieving:
    docs_scores = STORE.similarity_search_with_score(query_text, k=10)
    rows = []
    for doc, score in docs_scores:
        # We don't have 'id' persisted inside doc's metadata, so generate or extract
        _id = doc.metadata.get("id", str(uuid4()))
        rows.append([
            _id,
            doc.metadata.get("pdf_name"),
            doc.page_content,
            float(score)
        ])

    reranked = rerank_rows(query_text, rows, top_k=10)

    results = {
        "query": query_text,
        "results": [
            {
                "id":      r[0],
                "pdf":     r[1],
                "snippet": (r[2][:200] + "...") if len(r[2]) > 200 else r[2],
                "score":   r[3]
            }
            for r in reranked
        ]
    }
    results_json = json.dumps(results, indent=2)

    # combine context
    snippets = [r[2] for r in reranked]
    combined_context = "\n\n".join(f"Snippet {i+1}:\n{snip}" for i, snip in enumerate(snippets))

    return combined_context, results_json


# -----------------------------
# 5. Main chat function
# -----------------------------
def get_response_chat(query_text: str) -> Dict[str, Any]:
    combined_context, results_json = get_context(query_text)

    llm = ChatOpenAI(model_name=ANSWER_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY)

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{combined_context}

Question: {query_text}

Instructions:
1. Begin with a one- or two-sentence paragraph summarizing the main answer.
2. Then list any supporting details as bullet points.
3. Format your response in Markdown (paragraph, then '-' bullets).

Note: Only provide relevant answer, do not add extra information like what was present in the snippet and context.
Just provide the answer. 
Do not provide the info that how have you reached to the answer.
"""

    messages = [
        SystemMessage(content="Answer using ONLY the provided context. If missing, say you don't know."),
        HumanMessage(content=prompt)
    ]

    response = llm(messages)
    answer = response.content

    evaluation = evaluate_response(query_text, combined_context, answer)

    return {
        "answer": answer,
        "evaluation": evaluation,
        "retrieval": json.loads(results_json)
    }


# -----------------------------
# 6. Pretty printer that RETURNS (for Streamlit)
# -----------------------------
def pretty_print_response(resp: Dict[str, Any]) -> str:
    lines = []
    lines.append("=== Answer ===\n")
    lines.append(resp["answer"].strip() + "\n")

    lines.append("\n=== Retrieval Ranking ===\n")
    for i, item in enumerate(resp["retrieval"]["results"], 1):
        score = item.get("score")
        if score is None:
            lines.append(f"{i}. ID: {item['id']}, PDF: {item['pdf']}")
        else:
            lines.append(f"{i}. ID: {item['id']}, PDF: {item['pdf']}, Score: {score:.4f}")

    eval_ = resp["evaluation"]
    lines.append("\n=== Evaluation ===\n")
    lines.append(f"Rating: {eval_.get('rating')}/5")
    lines.append(f"Comments: {eval_.get('comments', '').strip()}")

    return "\n".join(lines)
