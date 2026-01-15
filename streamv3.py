import streamlit as st
import os, yaml
import uuid

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env from the current directory
except ImportError:
    # dotenv not installed, try manual loading
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
except Exception as e:
    # Silently fail if .env doesn't exist or has issues
    pass

from RagV2 import (
    RAG, list_devices, get_pipeline, get_model_list, get_llm, safe_idx
)
from ingest_safe import run_ingest
from defense.safe_retrieval import SafeIndex

# Initialize OpenTelemetry with Langfuse exporter
try:
    from observability.opentelemetry_integration import initialize_opentelemetry, OPENTELEMETRY_AVAILABLE
    if OPENTELEMETRY_AVAILABLE:
        # Initialize OpenTelemetry with Langfuse exporter
        otel_initialized = initialize_opentelemetry(
            service_name="rag-llm-system",
            service_version=os.getenv("APP_RELEASE", "1.0.0")
        )
        if otel_initialized:
            st.success("✅ OpenTelemetry + Langfuse observability enabled")
        else:
            st.info("ℹ️ OpenTelemetry not configured (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)")
    else:
        st.info("ℹ️ OpenTelemetry not available (install with: pip install opentelemetry-api opentelemetry-sdk langfuse)")
except Exception as e:
    st.warning(f"⚠️ OpenTelemetry initialization failed: {e}")
    # Fallback to direct Langfuse
    try:
        from observability.langfuse_integration import initialize_langfuse, LANGFUSE_AVAILABLE
        if LANGFUSE_AVAILABLE:
            langfuse_initialized = initialize_langfuse()
            if langfuse_initialized:
                st.info("ℹ️ Using direct Langfuse (OpenTelemetry unavailable)")
    except:
        pass

# Import enhanced structured guardrails
try:
    from nvidia_nemo.enhanced_guardrails import EnhancedStructuredGuardrails
    GUARDRAILS_AVAILABLE = True
except ImportError:
    try:
        from nvidia_nemo.structured_guardrails import StructuredGuardrails as EnhancedStructuredGuardrails
        GUARDRAILS_AVAILABLE = True
    except ImportError:
        GUARDRAILS_AVAILABLE = False
        st.warning("⚠️ Guardrails not available (optional feature)")

st.title("LLM + Secure RAG — Demo (Enovos/Encevo)")

# --- Policy controls ---
with open("policy.yaml", "r", encoding="utf-8") as f:
    policy = yaml.safe_load(f)

c1, c2, c3, c4 = st.columns(4)
with c1:
    mode = st.selectbox("Policy mode", ["off","monitor","strict"],
                        index=["off","monitor","strict"].index(policy.get("mode","monitor")))
with c2:
    safe_mode = st.checkbox("Safe mode (no tools; retrieval only)", value=policy.get("safe_mode", True))
with c3:
    cite_or_silent = st.checkbox("Cite-or-silent", value=policy.get("output",{}).get("cite_or_silent", True))
with c4:
    use_guardrails = st.checkbox("🛡️ Enable Guardrails", value=True, help="NVIDIA NeMo Guardrails protection")

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
    # Initialize enhanced structured guardrails if available
    if GUARDRAILS_AVAILABLE and use_guardrails:
        try:
            nemo_config_path = os.path.join("nvidia_nemo", "config")
            policy_matrix_path = os.path.join("nvidia_nemo", "policy_matrix.yml")
            st.session_state.guardrails = EnhancedStructuredGuardrails(
                st.session_state.rag_model,
                allowed_domains=["RAG", "embeddings", "retrieval", "documents", 
                               "machine learning", "AI", "natural language processing"],
                nemo_config_path=nemo_config_path if os.path.exists(nemo_config_path) else None,
                policy_matrix_path=policy_matrix_path if os.path.exists(policy_matrix_path) else None
            )
        except Exception as e:
            st.warning(f"Could not initialize guardrails: {e}")
            st.session_state.guardrails = None
    else:
        st.session_state.guardrails = None

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

# Initialize session ID for tracing
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.button("Clear chat history"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())  # New session

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask the assistant…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        # Use structured guardrails if enabled
        if use_guardrails and GUARDRAILS_AVAILABLE and st.session_state.get('guardrails'):
            # Generate user ID (could be from authentication)
            user_id = f"user_{st.session_state.session_id[:8]}"
            
            try:
                answ, guard_results, log_lines = st.session_state.guardrails.answer(
                    prompt, 
                    role="analyst",
                    user_id=user_id,
                    session_id=st.session_state.session_id,
                    trace_name="rag_query_with_guardrails"
                )
                
                # Debug: Check if answer is empty
                if not answ or answ.strip() == "":
                    st.warning("⚠️ Guardrails returned empty response. Check logs below.")
                    answ = "I apologize, but I couldn't generate a response. Please check the guardrails evaluation below for details."
            except Exception as e:
                st.error(f"❌ Error in guardrails: {e}")
                import traceback
                st.code(traceback.format_exc())
                # Fallback to direct RAG
                try:
                    answ = st.session_state.rag_model.answer(prompt, role="analyst")
                except Exception as e2:
                    answ = f"❌ Error: {str(e2)}"
            
            # Display guardrails status with structured format
            with st.expander("🛡️ Guardrails Evaluation", expanded=True):
                # Show all 5 guards
                for result in guard_results:
                    severity_color = {
                        "allowed": "✅",
                        "blocked": "🚫",
                        "review": "⚠️"
                    }
                    icon = severity_color.get(result.severity.value, "ℹ️")
                    
                    st.write(f"{icon} **{result.guard_name}** - Severity: `{result.severity.value.upper()}`")
                    st.write(f"   *{result.reason}*")
                    
                    if result.severity.value == "blocked":
                        st.error(f"🚫 BLOCKED: {result.reason}")
                    elif result.severity.value == "review":
                        st.warning(f"⚠️ REVIEW: {result.reason}")
                
                # Show log format
                st.divider()
                st.write("**Guard Logs (Exact Format):**")
                for log_line in log_lines:
                    st.code(log_line, language=None)
                
                # Summary
                blocked_count = sum(1 for r in guard_results if r.severity.value == "blocked")
                review_count = sum(1 for r in guard_results if r.severity.value == "review")
                allowed_count = sum(1 for r in guard_results if r.severity.value == "allowed")
                
                st.write(f"**Summary**: {blocked_count} blocked, {review_count} review, {allowed_count} allowed")
        else:
            # Fallback to direct RAG (with tracing)
            user_id = f"user_{st.session_state.session_id[:8]}"
            answ = st.session_state.rag_model.answer(
                prompt,
                role="analyst",
                user_id=user_id,
                session_id=st.session_state.session_id
            )
        
        st.markdown(answ)
    st.session_state.messages.append({"role":"assistant","content":answ})

st.caption("Transparency: Answers cite approved sources when used. If no relevant source exists, the assistant may answer generally (policy-controlled).")
if use_guardrails and GUARDRAILS_AVAILABLE:
    st.info("🛡️ **Guardrails Active**: Input validation, jailbreak detection, PII redaction, and citation enforcement are enabled.")
