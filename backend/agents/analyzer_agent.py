# import os
# from typing import Dict, Any, List
# from .base_agent import BaseAgent
# from ..llm_client import LLMClient


# class AnalyzerAgent(BaseAgent):
#     name = "analyzer"

#     def __init__(self, llm_client: LLMClient | None = None):
#         # Load provider + model from .env
#         self.llm = llm_client or LLMClient(
#             provider=os.getenv("LLM_PROVIDER", "mock"),
#             model=os.getenv("LLM_MODEL")
#         )

#     def _format_context(self, results: List[Dict[str, Any]]) -> str:
#         return "\n\n---\n\n".join([r["text"] for r in results])

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         question = input_data["question"]
#         results = input_data.get("results", [])
#         has_context = input_data.get("has_context", False)

#         # 💡 If NO document context → answer using pretrained LLM directly
#         if not has_context:
#             answer = self.llm.generate(
#                 question=question,
#                 context=""
#             )
#             return {
#                 "question": question,
#                 "answer": answer,
#                 "context": [],
#                 "mode": "direct_llm"
#             }

#         # 💡 If we DO have retrieved chunks → use RAG
#         context = self._format_context(results)

#         answer = self.llm.generate(
#             question=question,
#             context=context
#         )

#         return {
#             "question": question,
#             "answer": answer,
#             "context": results,
#             "mode": "rag"
#         }

#--------------------------------------------------------------------------------


# import os
# from typing import Dict, Any, List
# from .base_agent import BaseAgent
# from ..llm_client import LLMClient
# from ..llm_client import HistoryType


# class AnalyzerAgent(BaseAgent):
#     name = "analyzer"

#     def __init__(self, llm_client: LLMClient | None = None):
#         self.llm = llm_client or LLMClient(
#             provider=os.getenv("LLM_PROVIDER", "mock"),
#             model=os.getenv("LLM_MODEL"),
#         )

#     def _format_context(self, results: List[Dict[str, Any]]) -> str:
#         return "\n\n---\n\n".join(r["text"] for r in results)

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         question = input_data["question"]
#         results = input_data.get("results", [])
#         has_context = input_data.get("has_context", False)
#         history: HistoryType = input_data.get("history", [])

#         if not has_context:
#             # Direct LLM answer (no document context)
#             answer = self.llm.generate(question=question, context="", history=history)
#             updated_history = history + [{"user": question, "assistant": answer}]
#             return {
#                 "question": question,
#                 "answer": answer,
#                 "context": [],
#                 "mode": "direct_llm",
#                 "history": updated_history,
#             }

#         # RAG mode
#         context_str = self._format_context(results)
#         answer = self.llm.generate(question=question, context=context_str, history=history)
#         updated_history = history + [{"user": question, "assistant": answer}]

#         return {
#             "question": question,
#             "answer": answer,
#             "context": results,
#             "mode": "rag",
#             "history": updated_history,
#         }




# import os
# from typing import Dict, Any, List

# from .base_agent import BaseAgent
# from ..llm_client import LLMClient, HistoryType


# class AnalyzerAgent(BaseAgent):
#     name = "analyzer"

#     def __init__(self, llm_client: LLMClient | None = None):
#         self.llm = llm_client or LLMClient(
#             provider=os.getenv("LLM_PROVIDER", "mock"),
#             model=os.getenv("LLM_MODEL"),
#         )

#     def _format_context(self, items: List[Dict[str, Any]]) -> str:
#         return "\n\n---\n\n".join(entry["text"] for entry in items)

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         question = input_data["question"]

#         doc_results: List[Dict[str, Any]] = input_data.get("results", [])
#         mem_results: List[Dict[str, Any]] = input_data.get("memory_results", [])

#         has_context: bool = input_data.get("has_context", False)
#         has_memory: bool = input_data.get("has_memory", False)

#         history: HistoryType = input_data.get("history", [])

#         has_any = has_context or has_memory

#         # 🔸 No documents + no memory → pure LLM
#         if not has_any:
#             answer = self.llm.generate(
#                 question=question,
#                 context="",
#                 history=history,
#             )
#             updated_history = history + [{"user": question, "assistant": answer}]
#             return {
#                 "question": question,
#                 "answer": answer,
#                 "context": [],
#                 "mode": "direct_llm",
#                 "history": updated_history,
#             }

#         # 🔸 Build combined context (docs + memory)
#         parts: List[str] = []
#         if doc_results:
#             parts.append("DOCUMENT CONTEXT:\n" + self._format_context(doc_results))
#         if mem_results:
#             parts.append("MEMORY CONTEXT:\n" + self._format_context(mem_results))

#         context_str = "\n\n====\n\n".join(parts)

#         answer = self.llm.generate(
#             question=question,
#             context=context_str,
#             history=history,
#         )

#         updated_history = history + [{"user": question, "assistant": answer}]

#         combined_results = doc_results + mem_results

#         return {
#             "question": question,
#             "answer": answer,
#             "context": combined_results,
#             "mode": "rag",  # still "rag", but now includes memory
#             "history": updated_history,
#         }
#--------------------------------------------------------------------------------
#last

import os
from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..llm_client import LLMClient, HistoryType


class AnalyzerAgent(BaseAgent):
    name = "analyzer"

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient(
            provider=os.getenv("LLM_PROVIDER", "mock"),
            model=os.getenv("LLM_MODEL"),
        )

    def _format_context(self, items: List[Dict[str, Any]]) -> str:
        return "\n\n---\n\n".join(entry["text"] for entry in items)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        question = input_data["question"]

        doc_results: List[Dict[str, Any]] = input_data.get("results", [])
        mem_results: List[Dict[str, Any]] = input_data.get("memory_results", [])

        has_context: bool = input_data.get("has_context", False)
        has_memory: bool = input_data.get("has_memory", False)

        history: HistoryType = input_data.get("history", [])

        has_any = has_context or has_memory

        # ============================================================
        # 1️⃣ NO CONTEXT → allow normal conversational LLM
        # ============================================================
        if not has_any:
            answer = self.llm.generate(
                question=question,
                context="",
                history=history,  # full history is OK here
            )

            updated_history = history + [{"user": question, "assistant": answer}]
            return {
                "question": question,
                "answer": answer,
                "context": [],
                "mode": "direct_llm",
                "history": updated_history,
            }

        # ============================================================
        # 2️⃣ CONTEXT EXISTS → STRICT, ISOLATED RAG MODE
        # ============================================================

        # 🔒 STRICT grounding instruction
        grounding_instruction = (
            "You must answer the question using ONLY the information "
            "explicitly stated in the provided context below.\n\n"
            "If the answer is NOT present in the context, reply with:\n"
            "'The provided documents do not contain this information.'\n\n"
            "Do NOT use any external knowledge, assumptions, or prior training data."
        )

        # Build combined context (documents + memory)
        parts: List[str] = []
        if doc_results:
            parts.append("DOCUMENT CONTEXT:\n" + self._format_context(doc_results))
        if mem_results:
            parts.append("MEMORY CONTEXT:\n" + self._format_context(mem_results))

        context_str = grounding_instruction + "\n\n" + "\n\n====\n\n".join(parts)

        # 🔑 IMPORTANT FIX:
        # In RAG mode, we REMOVE assistant answers from history
        # to prevent self-reinforcement and answer leakage.
        rag_history: HistoryType = [
            turn for turn in history if "user" in turn
        ]

        answer = self.llm.generate(
            question=question,
            context=context_str,
            history=rag_history,  # user-only history
        )

        updated_history = history + [{"user": question, "assistant": answer}]
        combined_results = doc_results + mem_results

        return {
            "question": question,
            "answer": answer,
            "context": combined_results,
            "mode": "rag",  # grounded RAG (documents + memory)
            "history": updated_history,
        }

