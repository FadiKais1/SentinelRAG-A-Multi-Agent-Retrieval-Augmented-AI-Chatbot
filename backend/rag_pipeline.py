# from typing import Dict, Any
# from .vector_store import VectorStore
# from .document_loader import load_documents
# from .agents.orchestrator import Orchestrator

# _store: VectorStore | None = None
# _orch: Orchestrator | None = None

# def get_store() -> VectorStore:
#     global _store
#     if _store is None:
#         _store = VectorStore()
#     return _store

# def get_orchestrator() -> Orchestrator:
#     global _orch
#     if _orch is None:
#         _orch = Orchestrator()
#     return _orch

# def index_all_documents() -> int:
#     """
#     Loads all documents from /data/docs, chunks them,
#     and inserts them into the vector store.
#     """
#     docs = load_documents()
#     store = get_store()
#     store.add_documents(docs)
#     return len(docs)

# def answer_query(question: str) -> Dict[str, Any]:
#     orch = get_orchestrator()
#     return orch.answer_question(question)

#--------------------------------------------------------------------------------

# from typing import Dict, Any, List
# from .vector_store import VectorStore
# from .document_loader import load_documents
# from .agents.orchestrator import Orchestrator
# from .llm_client import HistoryType

# _store: VectorStore | None = None
# _orch: Orchestrator | None = None


# def get_store() -> VectorStore:
#     global _store
#     if _store is None:
#         _store = VectorStore()
#     return _store


# def get_orchestrator() -> Orchestrator:
#     global _orch
#     if _orch is None:
#         _orch = Orchestrator()
#     return _orch


# def index_all_documents() -> int:
#     docs = load_documents()
#     store = get_store()
#     store.add_documents(docs)
#     return len(docs)


# def answer_query(question: str, history: HistoryType | None = None) -> Dict[str, Any]:
#     orch = get_orchestrator()
#     return orch.answer_question(question, history=history)


from typing import Dict, Any

from .vector_store import VectorStore, MemoryStore
from .document_loader import load_documents
from .agents.orchestrator import Orchestrator
from .llm_client import HistoryType

_store: VectorStore | None = None
_orch: Orchestrator | None = None
_memory_store: MemoryStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


def index_all_documents() -> int:
    docs = load_documents()
    store = get_store()
    store.add_documents(docs)
    return len(docs)


def _should_remember(text: str) -> bool:
    """
    Very simple heuristic: automatically store sentences that look like 'facts'.
    (Option B: store many things, not only explicit 'remember' commands.)
    """
    lower = text.lower()
    if len(lower.split()) < 4:
        return False

    keywords = [" is ", " am ", " are ", " have ", " has ", " lives ", " love ", " likes ", " work ", " live "]
    return any(kw in lower for kw in keywords)


def remember_user_message(text: str) -> None:
    if not _should_remember(text):
        return

    mem_store = get_memory_store()
    mem_store.add_memory(
        text,
        metadata={"source": "memory", "kind": "user_fact"},
    )


def answer_query(question: str, history: HistoryType | None = None) -> Dict[str, Any]:
    """
    Top-level API used by the frontend.

    1. Optionally remembers the user question as a 'fact' (if heuristic says so).
    2. Calls Orchestrator to combine:
       - Document RAG
       - Long-term memory
       - LLM + conversation history
    """
    # 1️⃣ Store message into memory if it looks like a fact
    remember_user_message(question)

    # 2️⃣ Orchestrate retrieval + answering
    orch = get_orchestrator()
    return orch.answer_question(question, history=history or [])
