import streamlit as st
import os, yaml

from RagV2 import (
    RAG, list_devices, get_pipeline, get_model_list, get_llm, safe_idx
)
from ingest_safe import run_ingest
from defense.safe_retrieval import SafeIndex

st.title("LLM + Secure RAG — Demo (Enovos/Encevo)")

# --- Policy controls ---
with open("policy.yaml", "r", encoding="utf-8") as f:
    policy = yaml.safe_load(f)

c1, c2, c3 = st.columns(3)
with c1:
    mode = st.selectbox("Policy mode", ["off","monitor","strict"],
                        index=["off","monitor","strict"].index(policy.get("mode","monitor")))
with c2:
    safe_mode = st.checkbox("Safe mode (no tools; retrieval only)", value=policy.get("safe_mode", True))
with c3:
    cite_or_silent = st.checkbox("Cite-or-silent", value=policy.get("output",{}).get("cite_or_silent", True))

if st.button("Save policy"):
    policy["mode"] = mode
    policy["safe_mode"] = safe_mode
    policy.setdefault("output", {})["cite_or_silent"] = cite_or_silent
    with open("policy.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump(policy, f)
    st.success("policy.yaml saved.")

# --- Device selection ---
available_devices = list_devices()
selected_device = st.selectbox("Computation device", available_devices)
st.session_state.device = selected_device

# --- RAG model instance ---
@st.cache_resource
def _get_rag(method, device_sel):
    return RAG(method=method, device=device_sel)

mode_sel = st.selectbox('Select running mode', ['User BM25', 'Developer BERT'])
method = 'BM25' if mode_sel == 'User BM25' else st.selectbox('RAG methods', ['Default', 'BERT', 'BM25'])

if 'rag_model' not in st.session_state or st.session_state.get('name') != method:
    st.session_state.rag_model = _get_rag(method, selected_device)
    st.session_state.name = method

# --- LLM selection ---
llm_name = st.selectbox("Select LLM (or 'Other')", ['Please select LLM model'] + get_model_list())
if 'llm' not in st.session_state and llm_name != 'Please select LLM model':
    if llm_name == 'Other':
        llm_path = st.text_input('Provide Hugging Face model id')
        if llm_path:
            llm_name = get_llm(llm_path)
    st.session_state.rag_model.pipe = get_pipeline(st.session_state.rag_model.pipe_model)
    st.session_state.llm = True
    st.success(f"Model {llm_name} loaded")

st.divider()
st.subheader("1) Approved documents")

path_to_dir = st.text_input('Folder with raw docs (.pdf, .docx, .txt)', './database')
collection = st.text_input('Collection name', 'grid_ops')

c_ing1, c_ing2 = st.columns([1,1])
with c_ing1:
    if st.button("Run Safe Ingest"):
        res = run_ingest(src=path_to_dir, collection=collection)
        safe_idx.reload()  # refresh in-memory index
        st.success(f"Ingested {res['files']} files → {res['chunks']} chunks into ./safe_index")

with c_ing2:
    man = "./safe_index/manifest.json"
    if os.path.exists(man):
        st.download_button("Download manifest.json", open(man,"rb"), file_name="manifest.json")

_idx = SafeIndex()
st.info(f"Safe index status: {len(_idx.records)} chunks indexed.")

# Optional: legacy retriever population (keeps your previous flow)
if st.button("Process with legacy retriever (optional)"):
    try:
        from model.database import doc2Text
        data = doc2Text(path_to_dir)
        st.session_state.rag_model.model.process(doc=data, path=path_to_dir)
        st.session_state.rag_model.path = os.path.abspath(path_to_dir)
        st.success("Legacy retriever processed (BM25/BERT).")
    except Exception as e:
        st.error(f"Legacy process failed: {e}")

st.divider()
st.subheader("2) Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat history"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask the assistant…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        answ = st.session_state.rag_model.answer(prompt)
        st.markdown(answ)
    st.session_state.messages.append({"role":"assistant","content":answ})

st.caption("Transparency: Answers cite approved sources when used. If no relevant source exists, the assistant may answer generally (policy-controlled).")
