"""
NeMo Guardrails Custom Configuration
Initialization code for guardrails
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG components
try:
    from RagV2 import RAG, safe_idx
    from defense.guards import POLICY
except ImportError as e:
    print(f"Warning: Could not import RAG components: {e}")

# Global RAG instance (will be set by integration)
_rag_instance = None

def set_rag_instance(rag_instance):
    """Set the RAG instance for use by guardrails"""
    global _rag_instance
    _rag_instance = rag_instance

def get_rag_instance():
    """Get the RAG instance"""
    return _rag_instance

