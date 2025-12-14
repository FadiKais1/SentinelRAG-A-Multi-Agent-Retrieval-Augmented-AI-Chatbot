# import os
# from typing import Literal, Optional
# import requests


# class LLMClient:
#     def __init__(
#         self,
#         provider: Literal["mock", "nvidia", "openai"] = "mock",
#         model: Optional[str] = None
#     ):
#         self.provider = provider
#         self.model = model or os.getenv("LLM_MODEL")

#         # NVIDIA
#         self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
#         self.nvidia_base_url = os.getenv("NVIDIA_BASE_URL")

#         # OpenAI
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         self.openai_base_url = os.getenv("OPENAI_BASE_URL")

#     # -----------------------------
#     # MAIN DISPATCH
#     # -----------------------------
#     def generate(self, question: str, context: str) -> str:
#         if self.provider == "mock":
#             return self._mock(question, context)
#         if self.provider == "nvidia":
#             return self._nvidia(question, context)
#         if self.provider == "openai":
#             return self._openai(question, context)
#         return self._mock(question, context)

#     # -----------------------------
#     # MOCK (fallback)
#     # -----------------------------
#     def _mock(self, question: str, context: str) -> str:
#         size = len(context.split()) if context else 0
#         return (
#             f"[MOCK ANSWER]\n"
#             f"Q: {question}\n"
#             f"Context words: {size}\n"
#             f"(Connect real LLM in .env to replace this.)"
#         )

#     # -----------------------------
#     # NVIDIA IMPLEMENTATION
#     # -----------------------------
#     def _nvidia(self, question: str, context: str) -> str:
#         url = f"{self.nvidia_base_url}/chat/completions"
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.nvidia_api_key}"
#         }
#         prompt = (
#             "Answer the question using ONLY the provided context.\n\n"
#             f"QUESTION:\n{question}\n\n"
#             f"CONTEXT:\n{context}\n\nANSWER:"
#         )

#         payload = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.2
#         }

#         response = requests.post(url, json=payload, headers=headers)
#         result = response.json()

#         return result["choices"][0]["message"]["content"]

#     # -----------------------------
#     # OPENAI IMPLEMENTATION
#     # -----------------------------
#     def _openai(self, question: str, context: str) -> str:
#         url = f"{self.openai_base_url}/chat/completions"

#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.openai_api_key}"
#         }

#         payload = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": "Answer ONLY using the provided context."},
#                 {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}"}
#             ],
#             "temperature": 0.2
#         }

#         response = requests.post(url, json=payload, headers=headers)
#         result = response.json()

#         return result["choices"][0]["message"]["content"]


import os
from typing import Literal, Optional, List, Dict
import requests
from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


HistoryType = List[Dict[str, str]]  # [{"user": "...", "assistant": "..."}, ...]


class LLMClient:
    def __init__(
        self,
        provider: Literal["mock", "nvidia", "openai"] | None = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL

        self.nvidia_api_key = NVIDIA_API_KEY
        self.nvidia_base_url = NVIDIA_BASE_URL

        self.openai_api_key = OPENAI_API_KEY
        self.openai_base_url = OPENAI_BASE_URL

    # ----------------------------------------------------
    # Public entry
    # ----------------------------------------------------
    def generate(self, question: str, context: str, history: HistoryType | None = None) -> str:
        history = history or []

        if self.provider == "mock":
            return self._mock(question, context, history)

        if self.provider == "nvidia":
            return self._nvidia(question, context, history)

        if self.provider == "openai":
            return self._openai(question, context, history)

        return self._mock(question, context, history)

    # ----------------------------------------------------
    # Mock impl
    # ----------------------------------------------------
    def _mock(self, question: str, context: str, history: HistoryType) -> str:
        history_str = "\n".join(
            [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history]
        )

        return (
            f"[MOCK ANSWER]\n"
            f"Conversation so far:\n{history_str or '(no history)'}\n\n"
            f"Question: {question}\n"
            f"Context word count: {len(context.split()) if context else 0}\n"
            f"(Configure NVIDIA or OpenAI in .env to get real answers.)"
        )

    # ----------------------------------------------------
    # NVIDIA impl
    # ----------------------------------------------------
    def _nvidia(self, question: str, context: str, history: HistoryType) -> str:
        if not self.nvidia_api_key:
            return "NVIDIA_API_KEY is missing in .env."

        url = f"{self.nvidia_base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.nvidia_api_key}",
        }

        messages = []

        # System message: hybrid behavior
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "If CONTEXT is provided, rely strongly on it. "
                    "If CONTEXT is empty, answer using your own knowledge."
                ),
            }
        )

        # Add conversation history
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # Add new user prompt
        user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context or '(no context)'}"
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    # ----------------------------------------------------
    # OpenAI impl
    # ----------------------------------------------------
    def _openai(self, question: str, context: str, history: HistoryType) -> str:
        if not self.openai_api_key:
            return "OPENAI_API_KEY is missing in .env."

        url = f"{self.openai_base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        messages = []

        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "If CONTEXT is provided, rely strongly on it. "
                    "If CONTEXT is empty, answer using your own knowledge."
                ),
            }
        )

        # History
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # New question
        user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context or '(no context)'}"
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]
