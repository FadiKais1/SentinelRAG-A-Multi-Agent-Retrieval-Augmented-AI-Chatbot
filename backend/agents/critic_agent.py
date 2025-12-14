# import os
# from typing import Dict, Any
# from .base_agent import BaseAgent
# from ..llm_client import LLMClient


# class CriticAgent(BaseAgent):
#     name = "critic"

#     def __init__(self, llm_client: LLMClient | None = None):
#         self.llm = llm_client or LLMClient(
#             provider=os.getenv("LLM_PROVIDER", "mock"),
#             model=os.getenv("LLM_MODEL")
#         )

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         # Critic ONLY applies to RAG mode
#         if input_data.get("mode") != "rag":
#             return {
#                 "question": input_data["question"],
#                 "answer": input_data["answer"],
#                 "context": [],
#                 "critique": "No critique needed. Answered using general LLM.",
#                 "mode": "direct_llm"
#             }

#         question = input_data["question"]
#         answer = input_data["answer"]
#         context_chunks = input_data.get("context", [])

#         context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

#         critique = self.llm.generate(
#             question=f"Evaluate the following answer:\n\n{answer}\n\n",
#             context=f"Context:\n{context}"
#         )

#         return {
#             "question": question,
#             "answer": answer,
#             "critique": critique,
#             "context": context_chunks,
#             "mode": "rag"
#         }


import os
from typing import Dict, Any
from .base_agent import BaseAgent
from ..llm_client import LLMClient


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient(
            provider=os.getenv("LLM_PROVIDER", "mock"),
            model=os.getenv("LLM_MODEL"),
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if input_data.get("mode") != "rag":
            # No critique needed for pure LLM answers
            return {
                "question": input_data["question"],
                "answer": input_data["answer"],
                "context": [],
                "critique": "No critique. Answered using general LLM (no document context).",
                "mode": "direct_llm",
                "history": input_data.get("history", []),
            }

        question = input_data["question"]
        answer = input_data["answer"]
        context_chunks = input_data.get("context", [])
        history = input_data.get("history", [])

        context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)

        critique = self.llm.generate(
            question=f"Evaluate the following answer for correctness and faithfulness:\n\n{answer}",
            context=f"Context used to answer:\n\n{context_text}",
            history=history,
        )

        return {
            "question": question,
            "answer": answer,
            "critique": critique,
            "context": context_chunks,
            "mode": "rag",
            "history": history,
        }
