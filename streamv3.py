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
    # Cite-or-silent: ON = only answer from docs, OFF = always answer
    cite_or_silent = st.checkbox(
        "Cite-or-silent", 
        value=policy.get("output",{}).get("cite_or_silent", True),
        help="ON: Only answer if info found in documents. OFF: Always answer using LLM."
    )
    # Update in-memory policy immediately when checkbox changes
    from defense.guards import update_cite_or_silent
    update_cite_or_silent(cite_or_silent)
with c4:
    use_guardrails = st.checkbox("🛡️ Enable Guardrails", value=True, help="NVIDIA NeMo Guardrails protection")

if st.button("Save policy"):
    policy["mode"] = mode
    policy["safe_mode"] = safe_mode
    policy.setdefault("output", {})["cite_or_silent"] = cite_or_silent
    with open("policy.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump(policy, f)
    # Reload the in-memory POLICY so changes take effect immediately
    from defense.guards import reload_policy
    reload_policy()
    st.success("policy.yaml saved and applied.")

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

llm_name = st.selectbox("Select LLM (or 'Other')", all_models)

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
                        st.session_state.current_llm_name = llm_name
                        st.session_state.llm = True
                        st.success(f"✅ Connected to BeamStudio: {model}")
                    except Exception as e:
                        st.error(f"❌ Failed to connect to BeamStudio: {e}")
        elif llm_name == 'Other':
            llm_path = st.text_input('Provide Hugging Face model id')
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
                        # --- High-level row: Input Guards | LLM Generation | LLM Judge | Output Guards ---
                        input_guards_key = next((k for k in layers if 'parallel_input' in k or 'input_guards' in k.lower()), None)
                        output_guards_key = next((k for k in layers if 'parallel_output' in k or 'output_guards' in k.lower()), None)
                        llm_judge_key = next((k for k in layers if 'llm_judge' in k.lower()), None)
                        llm_gen_key = next((k for k in layers if 'llm_generation' in k.lower()), None)
                        
                        display_layers = []
                        if input_guards_key:
                            display_layers.append((input_guards_key, layers[input_guards_key]))
                        if llm_gen_key:
                            display_layers.append((llm_gen_key, layers[llm_gen_key]))
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
                        
                        # --- Deep dive: per-guard timings (always visible) ---
                        st.markdown("**📊 Per-guard breakdown**")
                        
                        # Input guards deep dive
                        if input_guards_key:
                            input_data = layers[input_guards_key]
                            details = input_data.get('details', {})
                            individual = details.get('individual_timings', {})
                            if individual:
                                # Sort by time descending to show bottleneck first
                                sorted_guards = sorted(individual.items(), key=lambda x: -x[1])
                                max_time = max(individual.values()) if individual else 0
                                # Human-readable guard names
                                guard_labels = {
                                    "embedding-similarity": "Embedding similarity",
                                    "llm-guard": "LLM Guard",
                                    "topic-taxonomy": "Topic taxonomy",
                                    "input-security": "NeMo (input guardrails 2 layers)",
                                    "input-sentimental": "Input sentimental",
                                    "input-topic": "Input topic",
                                }
                                bottleneck_shown = False
                                for guard_name, guard_time in sorted_guards:
                                    label = guard_labels.get(guard_name, guard_name.replace("-", " ").title())
                                    is_bottleneck = not bottleneck_shown and guard_time >= max_time and max_time > 0
                                    if is_bottleneck:
                                        bottleneck_shown = True
                                    delta = " ← bottleneck" if is_bottleneck else ""
                                    st.caption(f"• **{label}**: {guard_time:.0f}ms{delta}")
                            else:
                                st.caption(f"• _Input guards total: {input_data.get('duration_ms', 0):.0f}ms (no per-guard breakdown)_")
                        
                        # LLM Judge: show abortion/skip reason when skipped
                        if not llm_judge_key:
                            st.caption("• **LLM Judge**: skipped — no guard requested escalation → not invoked (saves ~1–2s)")
                        
                        # Output guards deep dive
                        if output_guards_key:
                            out_data = layers[output_guards_key]
                            details = out_data.get('details', {})
                            individual = details.get('individual_timings', {})
                            if individual:
                                sorted_guards = sorted(individual.items(), key=lambda x: -x[1])
                                out_labels = {
                                    "output-differential": "Output differential",
                                    "output-topic": "Output topic",
                                    "output-integrity": "Output integrity",
                                    "output-ip": "Output IP",
                                    "output-global": "Output global",
                                }
                                for guard_name, guard_time in sorted_guards:
                                    label = out_labels.get(guard_name, guard_name.replace("-", " ").title())
                                    st.caption(f"• **{label}**: {guard_time:.0f}ms")
                            else:
                                st.caption(f"• _Output guards total: {out_data.get('duration_ms', 0):.0f}ms_")
                    
                    st.divider()
                
                # Layer architecture visual
                st.subheader("🔄 Defense Pipeline (Speculative Parallel)")
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
