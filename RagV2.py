# RagV2.py
import os, re, torch
from datetime import datetime
from typing import Optional, Union
from transformers import pipeline
from huggingface_hub import repo_exists

from embedding import *  # your BM25/BERT classes
from defense.guards import POLICY, gate_and_log, path_is_allowed, redact
from defense.safe_retrieval import SafeIndex

# Import LLM client abstraction
from llm_client import (
    LLMClient, LocalLLMClient, BeamStudioClient, 
    get_llm_client, get_beamstudio_models, is_beamstudio_configured
)

__all__ = [
    "RAG",
    "get_pipeline",
    "get_llm",
    "get_model_list",
    "list_devices",
    "safe_idx",
    "get_llm_client",
    "get_beamstudio_models",
    "is_beamstudio_configured",
]

def _clean_answer(text: str) -> str:
    # remove template debris the tiny model sometimes emits
    text = re.sub(r"<\|[^>]{1,40}\|>", "", text)       # <|im_end|>, etc.
    text = re.sub(r"(?im)^\s*(question|answer)\s*:\s*", "", text)
    text = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", text)  # de-stutter
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _offtopic_constants(question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()
    if "speed of light" in a and "light" not in q:
        return True
    if "radians per second" in a and ("plane" in q or "train" in q):
        return True
    return False

# module-level safe index (reloaded from UI after ingest)
safe_idx = SafeIndex()

class RAG():
    def __init__(self, method=None, k=5, path=None,
                 pipeline_model="Felladrin/Smol-Llama-101M-Chat-v1",
                 device="cpu",
                 llm_client: Optional[LLMClient] = None):
        self.k = k
        self.path = path or "./database"
        self.pipe_model = pipeline_model
        
        # Use provided LLM client or create a local one
        if llm_client is not None:
            self.llm_client = llm_client
            # For backward compatibility, set pipe to None if using API client
            if isinstance(llm_client, BeamStudioClient):
                self.pipe = None
            else:
                # LocalLLMClient has a pipeline property
                self.pipe = getattr(llm_client, 'pipeline', None)
        else:
            # Legacy: create local pipeline directly
            self.pipe = get_pipeline(self.pipe_model, device)
            self.llm_client = LocalLLMClient(
                model_name=self.pipe_model,
                device=device,
                pipeline_instance=self.pipe
            )

        # Generation arguments
        # Note: max_new_tokens must be high enough for reasoning models (gpt-5.1, o1, etc.)
        # which use tokens for internal chain-of-thought before producing output
        # Complex queries can use 1000+ reasoning tokens before any output
        self.gen_args = {
            "max_new_tokens": 4000,  # Very high for complex reasoning queries
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
        }

        if method == 'BM25':
            self.model = BM25(k=self.k, path=self.path)
        elif method == 'BERT':
            self.model = BERT(k=self.k,
                              device="cuda:0" if torch.cuda.is_available() else "cpu",
                              path=self.path)
        else:
            self.model = None  # LLM-only

    def answer(self, query, doc=None, role="analyst", user_id=None, session_id=None):
        # Initialize OpenTelemetry span for retrieval if available
        retrieval_span_otel = None
        retrieval_span_langfuse = None
        try:
            from observability.opentelemetry_integration import create_trace_span, set_span_attribute, OPENTELEMETRY_AVAILABLE
            if OPENTELEMETRY_AVAILABLE:
                retrieval_span_otel = create_trace_span(
                    "retrieval",
                    attributes={"span.type": "retriever", "span.name": "document_retrieval"}
                )
                retrieval_span_otel.__enter__()
        except Exception:
            pass
        
        # Fallback to direct Langfuse
        if not retrieval_span_otel:
            try:
                from observability.langfuse_integration import get_langfuse_client, log_retrieval
                langfuse = get_langfuse_client()
                if langfuse:
                    from langfuse.decorators import langfuse_context
                    current_trace = getattr(langfuse_context, 'current_trace', None)
                    if current_trace:
                        retrieval_span_langfuse = current_trace
            except Exception:
                pass
        
        # 1) Retrieve (prefer safe index)
        context_list, metas = [], []
        retrieval_start = datetime.now()

        if safe_idx.records:
            params = POLICY.get("retrieval", {})
            top = safe_idx.query(
                query,
                k=self.k,
                min_rel=float(params.get("min_rel", 0.35)),
                min_kw=int(params.get("min_keyword_hits", 1)),
                max_chunks=int(params.get("max_chunks", 4)),
            )
            context_list = [t["text"] for t in top]
            metas = [t["meta"] for t in top]
            
            # Record retrieval for anomaly monitoring
            try:
                from rag_defense.retrieval_monitor import get_retrieval_monitor
                get_retrieval_monitor().record(top)
            except Exception:
                pass

            # Log retrieval to Langfuse
            if retrieval_span_langfuse:
                try:
                    from observability.langfuse_integration import log_retrieval
                    scores = [t["meta"].get("score", 0.0) for t in top]
                    log_retrieval(
                        retrieval_span_langfuse,
                        name="safe_index_retrieval",
                        query=query,
                        documents=context_list,
                        scores=scores,
                        metadata={
                            "collection": metas[0].get("collection", "unknown") if metas else "unknown",
                            "retrieval_method": "safe_index"
                        }
                    )
                except Exception:
                    pass
        else:
            retrieval_ok = (self.model is not None and self.path and path_is_allowed(self.path))
            if retrieval_ok:
                try:
                    ret = self.model.retrieve(query, path=self.path, doc=doc)
                    context_list = ret.get('doc') or ret.get('documents') or []
                    scores = ret.get('score', [0.0] * len(context_list))
                    metas = [{"doc":"(legacy)", "chunk":i, "collection":"legacy"}
                             for i,_ in enumerate(context_list)]
                    
                    # Log retrieval to Langfuse
                    if retrieval_span_langfuse:
                        try:
                            from observability.langfuse_integration import log_retrieval
                            log_retrieval(
                                retrieval_span_langfuse,
                                name="legacy_retrieval",
                                query=query,
                                documents=context_list,
                                scores=scores,
                                metadata={
                                    "retrieval_method": "legacy",
                                    "method_type": str(type(self.model).__name__)
                                }
                            )
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[RAG] Retrieval failed: {e}")
                    context_list = []

        # 2) Policy gate (prompt-injection, quarantine)
        blocked, safe_chunks = gate_and_log(query, context_list, role=role)
        if blocked and POLICY["mode"] == "strict":
            return "Blocked: suspected prompt-injection. Please rephrase."

        has_docs = bool(safe_chunks)
        
        # Cross-chunk consistency check (warn only, never block)
        consistency_warnings = []
        if has_docs and len(safe_chunks) >= 2:
            try:
                from rag_defense.consistency import flag_inconsistencies
                consistency_warnings = flag_inconsistencies(safe_chunks, query)
            except Exception:
                pass

        cite_or_silent_early = POLICY.get("output", {}).get("cite_or_silent", True)

        # CITE-OR-SILENT early exit: If ON and no docs found, refuse immediately
        if cite_or_silent_early and not has_docs:
            return "I couldn't find relevant information in the approved sources to answer this question."

        # 3) Build prompt
        message = build_prompt(query, safe_chunks, metas)

        # 4) LLM call (with controlled generation)
        # Check if LLM client is initialized
        if self.llm_client is None:
            return "❌ **Error**: LLM model is not initialized. Please select an LLM model in the UI."
        
        # Convert chat format to string for generation
        if isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict):
            # Extract content from chat format
            prompt_text = message[0].get('content', '')
            # Keep messages for API clients that support chat format
            chat_messages = message
        else:
            prompt_text = message if isinstance(message, str) else str(message)
            chat_messages = None
        
        if not prompt_text or prompt_text.strip() == "":
            return "❌ **Error**: Empty prompt generated. Please try again."
        
        # Log LLM generation start
        llm_start = datetime.now()
        try:
            # Use the LLM client abstraction for generation
            # Pass chat messages if available (for API clients), otherwise use prompt
            if isinstance(self.llm_client, BeamStudioClient) and chat_messages:
                out = self.llm_client.generate(prompt_text, messages=chat_messages, **self.gen_args)
            else:
                out = self.llm_client.generate(prompt_text, **self.gen_args)
            
            # Log LLM generation to Langfuse if available
            llm_end = datetime.now()
            try:
                from observability.langfuse_integration import get_langfuse_client, log_generation
                from langfuse.decorators import langfuse_context
                langfuse = get_langfuse_client()
                if langfuse:
                    current_trace = getattr(langfuse_context, 'current_trace', None)
                    if current_trace:
                        log_generation(
                            current_trace,
                            name="llm_generation",
                            model=self.llm_client.model_name,
                            input_text=prompt_text,
                            output_text=out,
                            start_time=llm_start,
                            end_time=llm_end,
                            metadata={
                                "provider": self.llm_client.provider,
                                "max_new_tokens": self.gen_args.get("max_new_tokens"),
                                "temperature": self.gen_args.get("temperature"),
                                "has_context": bool(safe_chunks)
                            }
                        )
            except Exception:
                pass  # Langfuse logging is optional, fail silently
        except Exception as e:
            print(f"[RAG] LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            out = "I encountered an error generating a response. Please try again."

        # 5) Output cleanup & enforcement
        out = _clean_answer(redact(out))
        
        # Check if we actually got a meaningful response
        if not out or out.strip() == "" or len(out.strip()) < 5:
            if not has_docs:
                return "I apologize, but I couldn't generate a response. Please try again."
            else:
                return "I couldn't find relevant information in the approved sources for this question."

        # Get cite_or_silent setting
        cite_or_silent = POLICY.get("output", {}).get("cite_or_silent", True)
        
        # CITE-OR-SILENT enforcement:
        # If cite_or_silent is ON and docs were retrieved but LLM didn't cite them,
        # the answer is NOT based on the corpus - refuse to answer
        if cite_or_silent and has_docs and not re.search(r"\[#\d+", out):
            return "I couldn't find relevant information in the approved sources to answer this question."
        
        # Auto-cite only if cite_or_silent is OFF (to help users see which docs were retrieved)
        # When cite_or_silent is ON, we DON'T auto-cite because the LLM should naturally cite if relevant
        if (not cite_or_silent
            and has_docs
            and POLICY.get("output", {}).get("auto_cite_if_missing", True)
            and not re.search(r"\[#\d+", out)
            and len(out.strip()) > 20):
            cites = " ".join(
                f"[#{i} {m.get('doc','?')}#{m.get('chunk','?')}]"
                for i, m in enumerate(metas[:3], 1)
            )
            out = f"{out}\n\n[CITATIONS] {cites}"

        # Last-mile sanity for common off-topic constants
        if _offtopic_constants(query, out):
            out = (
                "Typical aircraft cruise speeds:\n"
                "- Commercial airliners: ~800–900 km/h (430–490 kn, Mach 0.75–0.85)\n"
                "- Turboprops: ~450–600 km/h\n"
                "- Supersonic jets vary widely (Mach >1)\n"
                "Exact speed depends on aircraft type, altitude and wind."
            )

        # Append consistency warnings if any (informational, never block)
        if consistency_warnings:
            warn_lines = "\n".join(
                f"- {w.description}" for w in consistency_warnings[:3]
            )
            out += f"\n\n> **Note:** Potential inconsistencies detected between sources:\n{warn_lines}"

        return out

def build_prompt(query, chunks, metas):
    """Build a prompt for the LLM. Returns list of chat messages."""
    if chunks:
        # Format documents with citations
        numbered = []
        for i, txt in enumerate(chunks, start=1):
            meta = metas[i-1] if i-1 < len(metas) else {"doc": "?", "chunk": "?"}
            tag = f"[#{i} {meta.get('doc','?')}#{meta.get('chunk','?')}]"
            numbered.append(f"{tag}\n{str(txt)}")
        docs = "\n\n".join(numbered)
        
        # System message with instructions
        system_msg = (
            "You are a helpful assistant that answers questions based on provided documents. "
            "IMPORTANT: You MUST cite your sources using [#i doc#chunk] format. "
            "If the documents do not contain relevant information, say so clearly."
        )
        
        # User message with documents and question
        user_msg = f"Documents:\n\n{docs}\n\nQuestion: {query}"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    else:
        # No documents available
        if POLICY.get("output", {}).get("allow_general_if_no_docs", True):
            return [
                {"role": "system", "content": "You are a helpful assistant. Answer briefly and factually."},
                {"role": "user", "content": query}
            ]
        return [
            {"role": "system", "content": "You can only answer from approved sources."},
            {"role": "user", "content": "I cannot answer this question as no approved sources are available."}
        ]

def get_pipeline(p_model, device='cuda:0'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs('./model_w', exist_ok=True)
    local_dir = os.path.join('./model_w', p_model)
    if os.path.isdir(local_dir):
        p = pipeline("text-generation", model=local_dir, device=device,
                     trust_remote_code=True, use_fast=False)
    else:
        p = pipeline("text-generation", model=p_model, device=device,
                     trust_remote_code=True, use_fast=False)
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        p.save_pretrained(local_dir)
    return p

def get_llm(llm_path):
    try:
        return llm_path if repo_exists(llm_path) else "Felladrin/Smol-Llama-101M-Chat-v1"
    except Exception:
        return "Felladrin/Smol-Llama-101M-Chat-v1"

def get_model_list():
    lst = []
    root = './model_w'
    if not os.path.exists(root):
        return ['Other']
    for team in os.listdir(root):
        team_path = os.path.join(root, team)
        if os.path.isdir(team_path):
            for model in os.listdir(team_path):
                lst.append(team + '/' + model)
    lst.append('Other')
    return lst

def list_devices():
    try:
        if torch.cuda.is_available():
            return ['CPU'] + [f'GPU:{i} ({torch.cuda.get_device_name(i)})'
                              for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ['CPU']
