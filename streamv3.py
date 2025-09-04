import streamlit as st
import RagV2 as r  
from model.database import doc2Text 

st.title("Exploration of Llm Inference On Technical Topics")
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

available_devices = r.list_devices()
selected_device = st.selectbox("Choose your computation device", available_devices)

# Save selected device to session_state
st.session_state.device = selected_device
    
# --- Caching Functions ---
@st.cache_data(show_spinner="Extracting text from document...")
def cached_doc2text(path):
    return doc2Text(path)

@st.cache_data
def get_cached_rag_model(method):
    return r.RAG(method=method, device = selected_device)

@st.cache_data
def get_cached_pipeline(model_name):
    return r.get_pipeline(model_name)

# --- UI Controls ---
mode = st.selectbox('Select running mode', ['User', 'Developper'])

if mode == 'User':
    method = 'BM25'
else:
    method = st.selectbox('RAG methods', ['Default', 'BERT', 'BM25'])

# --- Load RAG Model ---
if 'rag_model' not in st.session_state or st.session_state.get('name') != method:
    st.session_state.rag_model = get_cached_rag_model(method)
    st.session_state.name = method

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LLM Selection ---
llm_name = st.selectbox("Select the LLM you want to use. Select 'Other' if you wish to provide your own", 
                        ['Please select LLM model'] + r.get_model_list())

if 'llm' not in st.session_state and llm_name != 'Please select LLM model':
    if llm_name == 'Other':
        llm_path = st.text_input('Please provide path to your LLM')
        if llm_path:
            llm_name = r.get_llm(llm_path)

    st.session_state.rag_model.pipe = get_cached_pipeline(st.session_state.rag_model.pipe_model)
    st.session_state.llm = True
    st.success(f"Model {llm_name} loaded!")

# --- Document Input ---
path_to_file = st.text_input('Provide the path to your files on the server (.pdf, .docx)')

if path_to_file and 'embed' not in st.session_state:
    try:
        data = cached_doc2text(path_to_file)
        st.session_state.rag_model.model.process(doc=data, path=path_to_file)
        st.session_state.rag_model.path = "./"+path_to_file
        st.session_state.embed = True
    except Exception as e:
        st.error(f"Failed to load document: {e}")

# --- Chat Interface ---
if 'embed' in st.session_state:
    if st.button("Clear chat history"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Please provide your question to the assistant")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            
            answ = st.session_state.rag_model.answer(prompt)
            st.markdown(answ)
        st.session_state.messages.append({"role": "assistant", "content": answ})