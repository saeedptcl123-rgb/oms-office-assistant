import os
import json

import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- Config ----------------
INDEX_DIR = "faiss_index"
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.jsonl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="OMS Policy Assistant", page_icon="📄", layout="centered")


# ---------------- Load resources (cached) ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        INDEX_DIR, _embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets. Add it in Streamlit secrets settings.")
        st.stop()
    return Groq(api_key=api_key)


embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
client = load_groq_client()


# ---------------- Retrieval ----------------
def retrieve_chunks(query, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    return results


def build_context(chunks):
    context_parts = []
    for i, c in enumerate(chunks):
        dept = c.metadata.get("department", "unknown")
        src = c.metadata.get("source_file", "unknown")
        context_parts.append(f"[Source {i+1} | {dept} / {src}]\n{c.page_content}")
    return "\n\n".join(context_parts)


def ask_groq(question, context):
    system_prompt = (
        "You are an OMS policy assistant. Answer the user's question using ONLY "
        "the provided policy excerpts below. If the answer isn't in the excerpts, "
        "say you don't have that information in the knowledge base. Be concise and clear."
    )
    user_prompt = f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------- UI ----------------
st.title("📄 OMS Policy Assistant")
st.caption("Ask a question about company policies. Answers are grounded in your OMS knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['department']}** / {s['source_file']}")

# Chat input
query = st.chat_input("Ask about an OMS policy...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer..."):
            chunks = retrieve_chunks(query)
            context = build_context(chunks)
            answer = ask_groq(query, context)

            sources = []
            seen = set()
            for c in chunks:
                dept = c.metadata.get("department", "unknown")
                src = c.metadata.get("source_file", "unknown")
                key = (dept, src)
                if key not in seen:
                    seen.add(key)
                    sources.append({"department": dept, "source_file": src})

        st.markdown(answer)
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- **{s['department']}** / {s['source_file']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
