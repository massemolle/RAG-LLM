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
    RAG, list_devices, get_pipeline, get_model_list, get_llm, safe_idx,
    get_llm_client, get_beamstudio_models, is_beamstudio_configured
)
from llm_client import LLMClient, LocalLLMClient, BeamStudioClient
from rag.ingest import run_ingest
from rag.safe_retrieval import SafeIndex
from rag.classification import DataClassification
from rag.index_versioning import list_versions, rollback_index

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

# Initialize session state for checkboxes (only once)
if "cite_or_silent" not in st.session_state:
    st.session_state.cite_or_silent = policy.get("output", {}).get("cite_or_silent", True)
if "safe_mode" not in st.session_state:
    st.session_state.safe_mode = policy.get("safe_mode", True)
if "guardrails_mode" not in st.session_state:
    st.session_state.guardrails_mode = policy.get("guardrails_mode", "complete")

GUARDRAILS_OPTIONS = [
    ("Off (no guardrails)", "off"),
    ("Classic (LLM judge only)", "classic"),
    ("Complete (full pipeline)", "complete"),
]
_guardrails_labels = [x[0] for x in GUARDRAILS_OPTIONS]
_guardrails_values = [x[1] for x in GUARDRAILS_OPTIONS]

def _auto_save_policy():
    """Persist current policy controls to policy.yaml and reload in-memory policy."""
    # Read latest widget values from session state (on_change fires after state update)
    _safe = st.session_state.get("safe_mode_checkbox", True)
    _cos = st.session_state.get("cite_or_silent_checkbox", True)
    _gr_label = st.session_state.get("guardrails_mode_select", "Complete (full pipeline)")
    _gr_value = _guardrails_values[_guardrails_labels.index(_gr_label)] if _gr_label in _guardrails_labels else "complete"

    # Update session state mirrors
    st.session_state.safe_mode = _safe
    st.session_state.cite_or_silent = _cos
    st.session_state.guardrails_mode = _gr_value

    # Write to disk
    try:
        with open("policy.yaml", "r", encoding="utf-8") as f:
            _pol = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _pol = {}
    _pol["safe_mode"] = _safe
    _pol.setdefault("output", {})["cite_or_silent"] = _cos
    _pol["guardrails_mode"] = _gr_value
    with open("policy.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_pol, f)

    # Reload in-memory policy
    try:
        from defense.guards import reload_policy, update_cite_or_silent
        reload_policy()
        update_cite_or_silent(_cos)
    except ImportError:
        pass

c1, c2, c3 = st.columns(3)
with c1:
    safe_mode = st.checkbox("Safe mode (no tools; retrieval only)", 
                            value=st.session_state.safe_mode,
                            key="safe_mode_checkbox",
                            on_change=_auto_save_policy,
                            help="When enabled, disables tool use and restricts the assistant to retrieval-only answers from the knowledge base.")
    st.session_state.safe_mode = safe_mode
with c2:
    cite_or_silent = st.checkbox(
        "Restrict to documents only",
        value=st.session_state.cite_or_silent,
        key="cite_or_silent_checkbox",
        on_change=_auto_save_policy,
        help="ON: Only answer from indexed documents — refuses if no relevant source is found. OFF: Answer any question, using documents when available."
    )
    st.session_state.cite_or_silent = cite_or_silent
    # Update in-memory policy immediately when checkbox changes
    from defense.guards import update_cite_or_silent
    update_cite_or_silent(cite_or_silent)
with c3:
    current_mode = st.session_state.guardrails_mode
    idx = _guardrails_values.index(current_mode) if current_mode in _guardrails_values else 2  # default Complete
    guardrails_mode_label = st.selectbox(
        "Guardrails",
        _guardrails_labels,
        index=idx,
        key="guardrails_mode_select",
        on_change=_auto_save_policy,
        help="Off: no guardrails, direct API call. Classic: 5 LLM judges for input/output safety. Complete: full multi-layer pipeline (embedding, LLM Guard, NeMo, LLM judges)."
    )
    st.session_state.guardrails_mode = _guardrails_values[_guardrails_labels.index(guardrails_mode_label)]

# --- Device selection ---
available_devices = list_devices()
selected_device = st.selectbox("Computation device", available_devices,
                               help="Select the hardware device for local model inference (CPU or CUDA GPU).")
st.session_state.device = selected_device

# --- RAG model instance ---
@st.cache_resource
def _get_rag(method, device_sel):
    return RAG(method=method, device=device_sel)

mode_sel = st.selectbox('Select running mode', ['User BM25', 'Developer BERT'],
                        help="User BM25: fast keyword-based retrieval (recommended). Developer BERT: semantic retrieval using BERT embeddings (slower).")
method = 'BM25' if mode_sel == 'User BM25' else st.selectbox('RAG methods', ['Default', 'BERT', 'BM25'],
                                                               help="The embedding method used to match your query against documents. BM25 is fastest; BERT uses neural embeddings.")

if 'rag_model' not in st.session_state or st.session_state.get('name') != method:
    st.session_state.rag_model = _get_rag(method, selected_device)
    st.session_state.name = method

# Dynamic guardrails init: create or clear based on guardrails_mode (runs every rerun)
guardrails_mode = st.session_state.guardrails_mode
if guardrails_mode == "off":
    st.session_state.guardrails = None
    if hasattr(st.session_state, 'guardrails_mode_initialized'):
        del st.session_state.guardrails_mode_initialized
elif guardrails_mode in ("classic", "complete") and st.session_state.get("rag_model") and GUARDRAILS_AVAILABLE:
    need_init = (
        st.session_state.get("guardrails") is None
        or getattr(st.session_state.get("guardrails"), "guardrails_mode", None) != guardrails_mode
    )
    if need_init:
        try:
            nemo_config_path = os.path.join("nvidia_nemo", "config")
            policy_matrix_path = os.path.join("nvidia_nemo", "policy_matrix.yml")
            st.session_state.guardrails = EnhancedStructuredGuardrails(
                st.session_state.rag_model,
                allowed_domains=["RAG", "embeddings", "retrieval", "documents",
                                 "machine learning", "AI", "natural language processing"],
                nemo_config_path=nemo_config_path if os.path.exists(nemo_config_path) else None,
                policy_matrix_path=policy_matrix_path if os.path.exists(policy_matrix_path) else None,
                mode=guardrails_mode
            )
            st.session_state.guardrails_mode_initialized = guardrails_mode
        except Exception as e:
            st.warning(f"Could not initialize guardrails: {e}")
            st.session_state.guardrails = None
else:
    if guardrails_mode in ("classic", "complete") and not st.session_state.get("rag_model"):
        st.session_state.guardrails = None

# --- LLM selection ---
# Build model list with BeamStudio options
local_models = get_model_list()
beamstudio_models = [f"BeamStudio: {m}" for m in get_beamstudio_models()]
all_models = ['Please select LLM model'] + beamstudio_models + local_models

# Check BeamStudio configuration status
beamstudio_configured = is_beamstudio_configured()
if beamstudio_configured:
    st.success("✅ BeamStudio API configured")
else:
    st.info("ℹ️ BeamStudio API not configured - set BEAMSTUDIO_API_KEY in .env for cloud models")

# Default to BeamStudio gpt-5.1 (index 1: first option is "Please select LLM model")
llm_name = st.selectbox("Select LLM (or 'Other')", all_models, index=1,
                        help="Choose the language model. BeamStudio models are cloud-hosted; 'Other' lets you specify a local HuggingFace model.")

if llm_name != 'Please select LLM model':
    # Check if selection changed
    current_llm = st.session_state.get('current_llm_name', None)
    
    if current_llm != llm_name:
        if llm_name.startswith('BeamStudio:'):
            # BeamStudio cloud model selected
            if not beamstudio_configured:
                st.error("❌ BeamStudio API key not configured. Please set BEAMSTUDIO_API_KEY in .env")
            else:
                model = llm_name.replace('BeamStudio: ', '')
                with st.spinner(f"Connecting to BeamStudio ({model})..."):
                    try:
                        llm_client = BeamStudioClient(model=model)
                        # Update RAG model with new LLM client
                        st.session_state.rag_model.llm_client = llm_client
                        st.session_state.rag_model.pipe = None  # No local pipeline for API
                        
                        # Adjust token limits based on model type
                        if 'gpt-5.1' in model or 'o1' in model or 'o3' in model:
                            st.session_state.rag_model.gen_args["max_new_tokens"] = 4000
                            st.info(f"ℹ️ {model} is a reasoning model – 4000 max tokens")
                        elif 'gpt-4o' in model:
                            st.session_state.rag_model.gen_args["max_new_tokens"] = 2000
                            st.info(f"ℹ️ {model} – 2000 max tokens")
                        else:
                            st.session_state.rag_model.gen_args["max_new_tokens"] = 3000
                            st.info(f"ℹ️ {model} – 3000 max tokens")
                        
                        st.session_state.current_llm_name = llm_name
                        st.session_state.llm = True
                        st.success(f"✅ Connected to BeamStudio: {model}")
                    except Exception as e:
                        st.error(f"❌ Failed to connect to BeamStudio: {e}")
        elif llm_name == 'Other':
            llm_path = st.text_input('Provide Hugging Face model id',
                                     help="Enter a HuggingFace model identifier, e.g. 'Qwen/Qwen2-0.5B'. The model will be downloaded on first use.")
            if llm_path:
                validated_name = get_llm(llm_path)
                with st.spinner(f"Loading local model: {validated_name}..."):
                    try:
                        llm_client = LocalLLMClient(model_name=validated_name, device=selected_device)
                        st.session_state.rag_model.llm_client = llm_client
                        st.session_state.rag_model.pipe = llm_client.pipeline
                        st.session_state.current_llm_name = validated_name
                        st.session_state.llm = True
                        st.success(f"✅ Loaded local model: {validated_name}")
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
        else:
            # Local model selected
            with st.spinner(f"Loading local model: {llm_name}..."):
                try:
                    llm_client = LocalLLMClient(model_name=llm_name, device=selected_device)
                    st.session_state.rag_model.llm_client = llm_client
                    st.session_state.rag_model.pipe = llm_client.pipeline
                    st.session_state.current_llm_name = llm_name
                    st.session_state.llm = True
                    st.success(f"✅ Loaded local model: {llm_name}")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")

# Show current model status
if st.session_state.get('current_llm_name'):
    current = st.session_state.current_llm_name
    provider = "BeamStudio (cloud)" if current.startswith('BeamStudio') else "Local (Hugging Face)"
    st.caption(f"**Current LLM**: {current} ({provider})")

st.divider()
st.subheader("1) Approved documents")

path_to_dir = st.text_input('Folder with raw docs (.pdf, .docx, .txt)', './rag/data',
                            help="Path to the folder containing your source documents (.pdf, .docx, .txt). These will be scanned, classified, and indexed.")
collection = st.text_input('Collection name', 'grid_ops',
                           help="A label for this set of documents in the index. Used to group and identify chunks from the same ingestion batch.")

# Data classification selector
classification_options = ["auto (folder-based)"] + [dc.value for dc in DataClassification]
selected_class = st.selectbox(
    "Data classification",
    classification_options,
    index=0,
    help="public / entity_internal / group_internal are allowed. classified / secret are REJECTED."
)
_explicit_class = None if selected_class.startswith("auto") else selected_class

c_ing1, c_ing2 = st.columns([1, 1])
with c_ing1:
    if st.button("Run Safe Ingest", help="Scan documents for threats (injections, PII, hidden content), classify them, and build a secure search index."):
        with st.spinner("Scanning and ingesting documents..."):
            res = run_ingest(
                src=path_to_dir,
                collection=collection,
                classification_level=_explicit_class,
            )
            safe_idx.reload()
        # Store results in session state so they survive reruns
        st.session_state["last_ingest_result"] = res
        st.success(
            f"Ingested **{res['files']}** files, **{res['chunks']}** chunks. "
            f"Rejected: {res['rejected']}, Quarantined: {res['quarantined']}."
        )

# Show scan details from last ingest (persists across reruns)
_last_ingest = st.session_state.get("last_ingest_result")
if _last_ingest and _last_ingest.get("scan_summary"):
    _summary = _last_ingest["scan_summary"]
    # Counters for the header
    _n_ok = sum(1 for s in _summary if s.get("status") == "ingested")
    _n_quar = sum(1 for s in _summary if s.get("status") == "quarantined")
    _n_rej = sum(1 for s in _summary if s.get("status") == "rejected")
    _header = f"Scan results: {_n_ok} ingested, {_n_quar} quarantined, {_n_rej} rejected"
    with st.expander(_header, expanded=False):
        for info in _summary:
            status = info.get("status", "?")
            fname = info.get("file", "?")
            cls = info.get("classification", "?")
            flags_detail = info.get("flags_detail", [])

            if status == "rejected":
                st.warning(f"**{fname}** [{cls}] — REJECTED: {info.get('reason', '')}")
            elif status == "quarantined":
                st.error(
                    f"**{fname}** [{cls}] — QUARANTINED "
                    f"({info.get('scan_blocks', 0)} block, "
                    f"{info.get('scan_warns', 0)} warn, "
                    f"{info.get('scan_info', 0)} info)"
                )
            elif status == "ingested":
                warns = info.get("scan_warns", 0)
                suffix = f" ({warns} warnings)" if warns else ""
                st.success(f"**{fname}** [{cls}] — ingested, {info.get('chunks', 0)} chunks{suffix}")
            elif status == "skipped_empty":
                st.caption(f"**{fname}** — skipped (empty)")

            # Show detailed flags if any
            if flags_detail:
                for fl in flags_detail:
                    sev = fl.get("severity", "?")
                    cat = fl.get("category", "?")
                    desc = fl.get("description", "?")
                    matched = fl.get("matched_text", "")
                    pname = fl.get("pattern_name", "")
                    if sev == "block":
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;:red[**BLOCK**] `{cat}` / `{pname}` — {desc}")
                    elif sev == "warn":
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;:orange[**WARN**] `{cat}` / `{pname}` — {desc}")
                    else:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;:blue[**INFO**] `{cat}` / `{pname}` — {desc}")
                    if matched:
                        st.code(matched, language=None)

        # Overall stats
        total_flags = sum(len(s.get("flags_detail", [])) for s in _summary)
        if total_flags:
            st.divider()
            st.caption(f"Total flags across all files: **{total_flags}**")

with c_ing2:
    man = "./rag/index/manifest.json"
    if os.path.exists(man):
        st.download_button("Download manifest.json", open(man, "rb"), file_name="manifest.json",
                           help="Download the ingestion manifest containing file list, chunk counts, scan results, and provenance metadata.")

_idx = SafeIndex()
st.info(f"Safe index: **{len(_idx.records)}** chunks indexed.")

# Index versioning
_versions = list_versions()
if _versions:
    with st.expander(f"Index versions ({len(_versions)} snapshots)", expanded=False):
        for v in _versions:
            label = v.get("label", "?")
            chunks = v.get("chunks", "?")
            created = v.get("created", "?")
            st.caption(f"**{label}** — {chunks} chunks — {created}")
        rb_label = st.selectbox("Rollback to version", [v.get("label", "") for v in _versions], key="rb_version",
                                help="Select a previous index snapshot to restore. The current index is backed up before rollback.")
        if st.button("Rollback", help="Restore the selected index version. Your current index is automatically saved as a new snapshot first."):
            if rollback_index(rb_label):
                safe_idx.reload()
                st.success(f"Rolled back to version {rb_label}. Index reloaded.")
            else:
                st.error(f"Rollback failed for version {rb_label}.")

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
        # Use structured guardrails if enabled (classic or complete mode)
        if guardrails_mode != "off" and GUARDRAILS_AVAILABLE and st.session_state.get('guardrails'):
            # Generate user ID (could be from authentication)
            user_id = f"user_{st.session_state.session_id[:8]}"
            
            try:
                # New API returns 4 values: response, guard_results, log_lines, timing_info
                result = st.session_state.guardrails.answer(
                    prompt, 
                    role="analyst",
                    user_id=user_id,
                    session_id=st.session_state.session_id,
                    trace_name="rag_query_with_guardrails"
                )
                
                # Handle both old (3-tuple) and new (4-tuple) return formats
                if len(result) == 4:
                    answ, guard_results, log_lines, timing_info = result
                else:
                    answ, guard_results, log_lines = result
                    timing_info = None
                
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
                    guard_results = []
                    log_lines = []
                    timing_info = None
                except Exception as e2:
                    answ = f"❌ Error: {str(e2)}"
                    guard_results = []
                    log_lines = []
                    timing_info = None
            
            # ========== TRANSPARENT GUARDRAILS VIEW ==========
            with st.expander("🛡️ Guardrails Evaluation (Transparent View)", expanded=True):
                # Timing metrics first (if available)
                if timing_info:
                    st.subheader("⏱️ Pipeline Timing")
                    
                    # Total time with color coding
                    total_ms = timing_info.get('total_ms', 0)
                    if total_ms < 1000:
                        st.success(f"**Total Time:** {total_ms:.1f}ms (Fast)")
                    elif total_ms < 2000:
                        st.info(f"**Total Time:** {total_ms:.1f}ms (Normal)")
                    else:
                        st.warning(f"**Total Time:** {total_ms:.1f}ms (Slow)")
                    
                    layers = timing_info.get('layers', {})
                    if layers:
                        # --- Resolve layer keys ---
                        input_guards_key = next((k for k in layers if 'parallel_input' in k or 'input_guards' in k.lower() or k == 'input_llm_judge'), None)
                        output_guards_key = next((k for k in layers if 'parallel_output' in k or 'output_guards' in k.lower() or k == 'output_llm_judge'), None)
                        if not input_guards_key:
                            input_guards_key = next((k for k in layers if 'input_classic' in k), None)
                        if not output_guards_key:
                            output_guards_key = next((k for k in layers if 'output_classic' in k), None)
                        llm_judge_key = next((k for k in layers if 'llm_judge' in k.lower()), None)
                        llm_gen_key = next((k for k in layers if 'llm_generation' in k.lower()), None)
                        is_classic = (st.session_state.get("guardrails_mode") == "classic") or (input_guards_key and "classic" in input_guards_key)
                        
                        # --- High-level row: Classic = 3 cols (Input, LLM, Output); Complete = 4 with LLM Judge ---
                        display_layers = []
                        if input_guards_key:
                            display_layers.append((input_guards_key, layers[input_guards_key]))
                        if llm_gen_key:
                            display_layers.append((llm_gen_key, layers[llm_gen_key]))
                        if not is_classic:
                            if llm_judge_key:
                                display_layers.append((llm_judge_key, layers[llm_judge_key]))
                            else:
                                display_layers.append(("layer_3_llm_judge", {"duration_ms": 0, "result": "SKIPPED", "skipped": True, "details": {}}))
                        if output_guards_key:
                            display_layers.append((output_guards_key, layers[output_guards_key]))
                        
                        cols = st.columns(len(display_layers))
                        for i, (layer_name, layer_data) in enumerate(display_layers):
                            with cols[i]:
                                duration = layer_data.get('duration_ms', 0)
                                result_str = layer_data.get('result', 'N/A')
                                skipped = layer_data.get('skipped', False)
                                display_name = layer_name.replace('_', ' ').replace('parallel ', '').title()
                                if 'llm' in layer_name.lower() and 'judge' in layer_name.lower() and skipped:
                                    st.metric(display_name, "0ms", "SKIPPED (no escalation)")
                                elif skipped:
                                    st.metric(display_name, f"{duration:.0f}ms", "SKIPPED")
                                elif result_str == "BLOCKED":
                                    st.metric(display_name, f"{duration:.0f}ms", "BLOCKED", delta_color="inverse")
                                elif result_str == "ESCALATE":
                                    st.metric(display_name, f"{duration:.0f}ms", "ESCALATE")
                                else:
                                    st.metric(display_name, f"{duration:.0f}ms", "OK")
                        
                        # --- Deep dive: per-guard timings ---
                        st.markdown("**📊 Per-guard breakdown**")
                        if is_classic:
                            guard_labels_input = {"input-sentimental": "Input sentimental", "input-security": "Input security", "input-topic": "Input topic"}
                            guard_labels_output = {"output-topic": "Output topic", "output-global": "Output global", "output-llm-guard": "Output LLM Guard", "output-prompt-leakage": "Prompt leakage"}
                        else:
                            guard_labels_input = {
                                "embedding-similarity": "Embedding similarity",
                                "llm-guard": "LLM Guard",
                                "topic-taxonomy": "Topic taxonomy",
                                "input-security": "NeMo (input guardrails 2 layers)",
                                "input-sentimental": "Input sentimental",
                                "input-topic": "Input topic",
                            }
                            guard_labels_output = {
                                "output-differential": "Output differential",
                                "output-topic": "Output topic",
                                "output-integrity": "Output integrity",
                                "output-ip": "Output IP",
                                "output-global": "Output global",
                                "output-llm-guard": "Output LLM Guard",
                                "output-prompt-leakage": "Prompt leakage",
                            }
                        
                        # Input guards deep dive
                        if input_guards_key:
                            input_data = layers[input_guards_key]
                            details = input_data.get('details', {})
                            individual = details.get('individual_timings', {})
                            if individual:
                                sorted_guards = sorted(individual.items(), key=lambda x: -x[1])
                                max_time = max(individual.values()) if individual else 0
                                bottleneck_shown = False
                                for guard_name, guard_time in sorted_guards:
                                    label = guard_labels_input.get(guard_name, guard_name.replace("-", " ").title())
                                    is_bottleneck = not bottleneck_shown and guard_time >= max_time and max_time > 0
                                    if is_bottleneck:
                                        bottleneck_shown = True
                                    delta = " ← bottleneck" if is_bottleneck else ""
                                    st.caption(f"• **{label}**: {guard_time:.0f}ms{delta}")
                            else:
                                st.caption(f"• _Input guards total: {input_data.get('duration_ms', 0):.0f}ms (no per-guard breakdown)_")
                        
                        if not is_classic and not llm_judge_key:
                            st.caption("• **LLM Judge**: skipped — no guard requested escalation → not invoked (saves ~1–2s)")
                        
                        # Token usage from LLM (if available)
                        if hasattr(st.session_state, 'rag_model') and hasattr(st.session_state.rag_model, 'llm_client'):
                            client = st.session_state.rag_model.llm_client
                            if hasattr(client, 'last_usage') and client.last_usage:
                                usage = client.last_usage
                                st.markdown("**🔢 LLM Token Usage**")
                                token_cols = st.columns(4)
                                with token_cols[0]:
                                    st.metric("Prompt", f"{usage.get('prompt_tokens', 0):,}")
                                with token_cols[1]:
                                    st.metric("Completion", f"{usage.get('completion_tokens', 0):,}")
                                with token_cols[2]:
                                    reasoning = usage.get('reasoning_tokens', 0)
                                    st.metric("Reasoning", f"{reasoning:,}" if reasoning else "N/A")
                                with token_cols[3]:
                                    st.metric("Total", f"{usage.get('total_tokens', 0):,}")
                                
                                # Efficiency indicator
                                completion = usage.get('completion_tokens', 0)
                                reasoning = usage.get('reasoning_tokens', 0)
                                if completion > 0 and reasoning > 0:
                                    output_tokens = completion - reasoning
                                    efficiency = (output_tokens / completion * 100) if completion > 0 else 0
                                    st.caption(f"📈 Output efficiency: {efficiency:.0f}% ({output_tokens} output tokens / {completion} completion tokens)")
                                
                                finish = usage.get('finish_reason', 'unknown')
                                if finish == 'length':
                                    st.warning(f"⚠️ Finish reason: `{finish}` — response may have been truncated")
                                else:
                                    st.caption(f"Finish reason: `{finish}`")
                        
                        # Output guards deep dive
                        if output_guards_key:
                            out_data = layers[output_guards_key]
                            details = out_data.get('details', {})
                            individual = details.get('individual_timings', {})
                            if individual:
                                sorted_guards = sorted(individual.items(), key=lambda x: -x[1])
                                for guard_name, guard_time in sorted_guards:
                                    label = guard_labels_output.get(guard_name, guard_name.replace("-", " ").title())
                                    st.caption(f"• **{label}**: {guard_time:.0f}ms")
                            else:
                                st.caption(f"• _Output guards total: {out_data.get('duration_ms', 0):.0f}ms_")
                    
                    st.divider()
                
                # Layer architecture visual (classic vs complete)
                st.subheader("🔄 Defense Pipeline (Speculative Parallel)")
                if st.session_state.get("guardrails_mode") == "classic":
                    st.markdown("""
                    ```
                    Line A:  Query ─► PARALLEL 3 input LLM judges (sentimental, security, topic) ─► input_ok?
                    Line B:  Query ─► LLM (RAG or direct) ─► PARALLEL 2 output LLM judges (topic, global) ─► response_ok?
                    Display answer only if input_ok and response_ok (both lines run in parallel)
                    ```
                    """)
                else:
                    st.markdown("""
                    ```
                    Line A:  Query ─► PARALLEL input guards ─► [LLM Judge*] ─► input_ok?
                                    ┌─ Embedding, LLM Guard, NeMo, Topic, Sentimental, Input Topic
                                    └─ *only if escalated
                    Line B:  Query ─► LLM (RAG or direct) ─► PARALLEL output guards ─► response_ok?
                    Display answer only if input_ok and response_ok (both lines run in parallel)
                    ```
                    """)
                
                st.divider()
                
                # Guard results by category
                st.subheader("🔍 Guard Results")
                
                # Group guards by type
                input_guard_names = ['embedding-similarity', 'llm-guard', 'llm-guard-lite', 'topic-taxonomy', 'input-security', 'input-sentimental', 'input-topic', 'llm-judge', 'fast-path']
                input_guards = [r for r in guard_results if 'input' in r.guard_name.lower() or r.guard_name in input_guard_names]
                output_guards = [r for r in guard_results if 'output' in r.guard_name.lower()]
                other_guards = [r for r in guard_results if r not in input_guards and r not in output_guards]
                
                # Input guards
                if input_guards:
                    st.write("**Input Guards:**")
                    for result in input_guards:
                        severity_icon = {"allowed": "✅", "blocked": "🚫", "review": "⚠️", "escalate": "📤"}.get(result.severity.value, "ℹ️")
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.write(f"{severity_icon} `{result.severity.value.upper()}`")
                            with col2:
                                st.write(f"**{result.guard_name}**")
                                st.caption(result.reason[:200] + "..." if len(result.reason) > 200 else result.reason)
                
                # Output guards
                if output_guards:
                    st.write("**Output Guards:**")
                    for result in output_guards:
                        severity_icon = {"allowed": "✅", "blocked": "🚫", "review": "⚠️"}.get(result.severity.value, "ℹ️")
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.write(f"{severity_icon} `{result.severity.value.upper()}`")
                            with col2:
                                st.write(f"**{result.guard_name}**")
                                st.caption(result.reason[:200] + "..." if len(result.reason) > 200 else result.reason)
                
                # Other guards
                if other_guards:
                    st.write("**Other Guards:**")
                    for result in other_guards:
                        severity_icon = {"allowed": "✅", "blocked": "🚫", "review": "⚠️"}.get(result.severity.value, "ℹ️")
                        st.write(f"{severity_icon} **{result.guard_name}** - `{result.severity.value.upper()}`")
                        st.caption(result.reason)
                
                st.divider()
                
                # Summary metrics
                st.subheader("📊 Summary")
                blocked_count = sum(1 for r in guard_results if r.severity.value == "blocked")
                review_count = sum(1 for r in guard_results if r.severity.value == "review")
                allowed_count = sum(1 for r in guard_results if r.severity.value == "allowed")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Blocked", blocked_count, delta=None if blocked_count == 0 else "ALERT", delta_color="inverse" if blocked_count > 0 else "off")
                with col2:
                    st.metric("Review", review_count)
                with col3:
                    st.metric("Allowed", allowed_count)
                
                # Detailed logs (collapsed by default)
                with st.expander("📜 Detailed Guard Logs"):
                    for log_line in log_lines:
                        st.code(log_line, language=None)
            
            # Show response time prominently (outside expander)
            if timing_info:
                _total_ms = timing_info.get('total_ms', 0)
                st.caption(f"⏱️ **Response time:** {_total_ms:.0f} ms")
        else:
            # Guardrails off: direct RAG, still show time and token usage
            # Global rate limit check (even without guardrails)
            _rate_limited = False
            try:
                from nvidia_nemo.production_hardening import get_global_rate_limiter
                _rl_ok, _rl_reason = get_global_rate_limiter().check_global_limit()
                if not _rl_ok:
                    _rate_limited = True
                    answ = "Rate limit exceeded. Please wait before sending another query."
                    st.warning(f"⚠️ {_rl_reason}")
            except ImportError:
                pass

            user_id = f"user_{st.session_state.session_id[:8]}"
            import time as _time
            _t0 = _time.perf_counter()
            if not _rate_limited:
                answ = st.session_state.rag_model.answer(
                    prompt,
                    role="analyst",
                    user_id=user_id,
                    session_id=st.session_state.session_id
                )
            _elapsed_ms = (_time.perf_counter() - _t0) * 1000
            # Show response time prominently
            st.caption(f"⏱️ **Response time:** {_elapsed_ms:.0f} ms")
            with st.expander("Token usage details", expanded=False):
                if hasattr(st.session_state.rag_model, "llm_client") and getattr(st.session_state.rag_model.llm_client, "last_usage", None):
                    usage = st.session_state.rag_model.llm_client.last_usage
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Prompt tokens", f"{usage.get('prompt_tokens', 0):,}")
                    with c2:
                        st.metric("Completion tokens", f"{usage.get('completion_tokens', 0):,}")
                    with c3:
                        r = usage.get("reasoning_tokens", 0)
                        st.metric("Reasoning tokens", f"{r:,}" if r else "N/A")
                    with c4:
                        st.metric("Total tokens", f"{usage.get('total_tokens', 0):,}")
                else:
                    st.caption("Token usage not available (e.g. local model).")
        
        st.markdown(answ)
    st.session_state.messages.append({"role":"assistant","content":answ})

st.caption("Transparency: Answers cite approved sources when used. If 'Restrict to documents only' is off, the assistant may answer generally.")
