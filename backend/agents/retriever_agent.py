# from typing import Dict, Any
# from .base_agent import BaseAgent
# from ..vector_store import VectorStore


# class RetrieverAgent(BaseAgent):
#     name = "retriever"

#     def __init__(self, k: int = 5):
#         self.store = VectorStore()
#         self.k = k

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         question = input_data["question"]
#         k = input_data.get("k", self.k)

#         results = self.store.search(question, k=k)

#         # If nothing is retrieved → treat as "no context"
#         if not results:
#             return {
#                 "question": question,
#                 "results": [],
#                 "has_context": False
#             }

#         return {
#             "question": question,
#             "results": results,
#             "has_context": True
#         }

#-----------------------------------------------------------------


# from typing import Dict, Any
# from .base_agent import BaseAgent
# from ..vector_store import VectorStore


# class RetrieverAgent(BaseAgent):
#     name = "retriever"

#     def __init__(self, k: int = 5):
#         self.store = VectorStore()
#         self.k = k

#     def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         question = input_data["question"]
#         k = input_data.get("k", self.k)

#         results = self.store.search(question, k=k)

#         if not results:
#             return {
#                 "question": question,
#                 "results": [],
#                 "has_context": False,
#             }

#         return {
#             "question": question,
#             "results": results,
#             "has_context": True,
#         }
#-----------------------------------------------------------------
#last version
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..vector_store import VectorStore


class RetrieverAgent(BaseAgent):
    name = "retriever"

    def __init__(self, k: int = 5):
        self.store = VectorStore()
        self.k = k

    # ------------------------------------------------------------
    # Utility: extract a "main entity" or key phrase from question
    # ------------------------------------------------------------
    def _extract_entity(self, question: str) -> str:
        """
        Extract the main entity or subject from a question.
        Works for:
        - Who is X?
        - Tell me about X
        - Provide information about X
        Falls back gracefully for general questions.
        """
        q = question.lower()

        stop_phrases = [
            "who is",
            "who was",
            "tell me about",
            "provide information about",
            "can you",
            "please",
            "explain",
            "describe",
            "a person called",
            "what is",
            "what are",
            "give me information about",
        ]

        for sp in stop_phrases:
            q = q.replace(sp, "")

        return q.strip(" ?.,:")

    # ------------------------------------------------------------
    # Utility: boost chunks that explicitly mention the entity
    # ------------------------------------------------------------
    def _boost_entity_matches(
        self,
        results: List[Dict[str, Any]],
        entity: str,
    ) -> List[Dict[str, Any]]:
        if not entity:
            return results

        entity = entity.lower()

        boosted: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []

        for r in results:
            text = r.get("text", "").lower()
            if entity in text:
                boosted.append(r)
            else:
                others.append(r)

        return boosted + others

    # ------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        question = input_data["question"]
        k = input_data.get("k", self.k)

        # 1️⃣ Vector similarity search
        results = self.store.search(question, k=k)

        if not results:
            return {
                "question": question,
                "results": [],
                "has_context": False,
                "context_matches_query": False,
            }

        # 2️⃣ Extract entity / main subject
        entity = self._extract_entity(question)

        # 3️⃣ Boost results that explicitly mention entity
        results = self._boost_entity_matches(results, entity)

        # 4️⃣ Decide if context is actually relevant
        # Rule:
        # - If entity exists and appears in any chunk → relevant
        # - If entity is empty (general question) → assume relevant
        if entity:
            context_matches_query = any(
                entity in r.get("text", "").lower() for r in results
            )
        else:
            # General questions like "what is machine learning?"
            context_matches_query = True

        return {
            "question": question,
            "results": results,
            "has_context": True,
            "context_matches_query": context_matches_query,
        }
