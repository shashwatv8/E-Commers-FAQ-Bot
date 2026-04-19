
import streamlit as st
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from agent import get_app, DOCUMENTS, CapstoneState
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="E-Commerce FAQ Bot", layout="centered")
st.title(" E-Commerce FAQ Bot")

@st.cache_resource
def load_agent_and_kb():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=[d["id"] for d in DOCUMENTS], metadatas=[{"topic":d["topic"]} for d in DOCUMENTS])

    app = get_app(collection=collection, embedder=embedder)
    return app, collection

app, collection = load_agent_and_kb()

if "messages" not in st.session_state: st.session_state.messages = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write("Automated assistant for shipping, returns, and product inquiries.")
    st.subheader("Topics Covered")
    for d in DOCUMENTS:
        st.write(f" {d['topic']}") 
    st.divider()
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        res = app.invoke({"question":prompt}, config={"configurable":{"thread_id":st.session_state.thread_id}})
        ans = res.get("answer", "Error")
        st.write(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})
