# from typing import Dict, Any
# from .retriever_agent import RetrieverAgent
# from .analyzer_agent import AnalyzerAgent
# from .critic_agent import CriticAgent


# class Orchestrator:
#     """
#     Multi-agent pipeline:
#       1. RetrieverAgent → find context
#       2. AnalyzerAgent  → answer with RAG or direct LLM
#       3. CriticAgent    → critique only if RAG is used
#     """

#     def __init__(self):
#         self.retriever = RetrieverAgent(k=5)
#         self.analyzer = AnalyzerAgent()
#         self.critic = CriticAgent()

#     def answer_question(self, question: str) -> Dict[str, Any]:
#         # Step 1 — Retrieve context
#         retrieval_output = self.retriever.run({"question": question})

#         # Step 2 — Generate answer (RAG or direct LLM)
#         analysis_output = self.analyzer.run(retrieval_output)

#         # Step 3 — Critique only for RAG mode
#         if analysis_output.get("mode") == "rag":
#             return self.critic.run(analysis_output)

#         # Direct LLM answers → no critique
#         return {
#             "question": question,
#             "answer": analysis_output["answer"],
#             "context": [],
#             "critique": "Answered using general LLM (no document context found).",
#             "mode": "direct_llm"
#         }

#------------------------------------------------------------------------


# from typing import Dict, Any, List
# from .retriever_agent import RetrieverAgent
# from .analyzer_agent import AnalyzerAgent
# from .critic_agent import CriticAgent
# from ..llm_client import HistoryType


# class Orchestrator:
#     """
#     Multi-agent workflow:
#       1. RetrieverAgent  -> retrieves document chunks
#       2. AnalyzerAgent   -> answers (RAG or direct LLM)
#       3. CriticAgent     -> critiques only RAG answers
#     """

#     def __init__(self):
#         self.retriever = RetrieverAgent(k=5)
#         self.analyzer = AnalyzerAgent()
#         self.critic = CriticAgent()

#     def answer_question(self, question: str, history: HistoryType | None = None) -> Dict[str, Any]:
#         history = history or []

#         # Step 1: retrieval
#         retrieval_output = self.retriever.run({"question": question})
#         retrieval_output["history"] = history

#         # Step 2: analyzer (decides RAG vs direct LLM)
#         analysis_output = self.analyzer.run(retrieval_output)

#         # Step 3: critique only when in RAG mode
#         if analysis_output.get("mode") == "rag":
#             crit = self.critic.run(analysis_output)
#             return crit

#         # Direct LLM path
#         return analysis_output


from typing import Dict, Any

from .retriever_agent import RetrieverAgent
from .analyzer_agent import AnalyzerAgent
from .critic_agent import CriticAgent
from ..vector_store import MemoryStore
from ..llm_client import HistoryType


class Orchestrator:
    """
    Multi-agent workflow:
      1. RetrieverAgent  -> retrieves document chunks
      2. MemoryStore     -> retrieves long-term user memory
      3. AnalyzerAgent   -> answers (RAG + memory, or direct LLM)
      4. CriticAgent     -> critiques only RAG answers
    """

    def __init__(self):
        self.retriever = RetrieverAgent(k=5)
        self.analyzer = AnalyzerAgent()
        self.critic = CriticAgent()
        self.memory_store = MemoryStore()

    def answer_question(self, question: str, history: HistoryType | None = None) -> Dict[str, Any]:
        history = history or []

        # 1️⃣ Retrieve from documents
        retrieval_output = self.retriever.run({"question": question})

        # 2️⃣ Retrieve from memory
        mem_results = self.memory_store.search(question, k=5)
        has_memory = len(mem_results) > 0

        # 3️⃣ Analyzer gets both doc + memory + history
        analyzer_input: Dict[str, Any] = {
            "question": question,
            "results": retrieval_output.get("results", []),
            "memory_results": mem_results,
            "has_context": retrieval_output.get("has_context", False),
            "has_memory": has_memory,
            "history": history,
        }

        analysis_output = self.analyzer.run(analyzer_input)

        # 4️⃣ Critic only for RAG mode (docs/memory used)
        if analysis_output.get("mode") == "rag":
            crit = self.critic.run(analysis_output)
            return crit

        # Otherwise direct LLM
        return analysis_output
