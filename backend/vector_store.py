# import chromadb
# from chromadb.config import Settings
# from chromadb.api.types import Documents, Embeddings
# from chromadb.utils import embedding_functions

# from typing import List, Dict, Any
# from config import VECTOR_DB_DIR
# from .embeddings import embed_texts


# # Create a Chroma-compatible embedding wrapper
# class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
#     def __call__(self, inputs: Documents) -> Embeddings:
#         # inputs is a list[str]
#         return embed_texts(inputs)


# class VectorStore:
#     def __init__(self, collection_name: str = "documents"):
#         self.client = chromadb.PersistentClient(
#             path=str(VECTOR_DB_DIR),
#             settings=Settings(allow_reset=False)
#         )

#         self.embedding_fn = LocalEmbeddingFunction()

#         self.collection = self.client.get_or_create_collection(
#             name=collection_name,
#             embedding_function=self.embedding_fn
#         )

#     def add_documents(self, docs: List[Dict[str, Any]]):
#         self.collection.add(
#             ids=[d["id"] for d in docs],
#             documents=[d["text"] for d in docs],
#             metadatas=[d.get("metadata", {}) for d in docs],
#         )

#     def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
#         result = self.collection.query(
#             query_texts=[query],
#             n_results=k
#         )

#         docs = []
#         for i in range(len(result["ids"][0])):
#             docs.append({
#                 "id": result["ids"][0][i],
#                 "text": result["documents"][0][i],
#                 "metadata": result["metadatas"][0][i],
#                 "distance": result["distances"][0][i] if "distances" in result else None,
#             })

#         return docs





from __future__ import annotations

from typing import List, Dict, Any
import uuid

import chromadb
from chromadb.api.types import EmbeddingFunction

from config import VECTOR_DB_DIR
from .embeddings import embed_texts


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function wrapping our embed_texts() helper.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        return embed_texts(input)


class VectorStore:
    """
    Vector store for DOCUMENTS.
    Uses Chroma persistent client and a single collection 'documents'.
    """

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(),
        )

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """
        docs: list of dicts:
          {
            "id": str,
            "text": str,
            "metadata": { ... }
          }
        """
        if not docs:
            return

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        result = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        documents = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            out.append(
                {
                    "id": ids[i],
                    "text": doc,
                    "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                }
            )
        return out


class MemoryStore:
    """
    Separate collection for LONG-TERM MEMORY.
    Stores user-provided sentences and retrieves them by semantic similarity.
    """

    def __init__(self, collection_name: str = "memory"):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(),
        )

    def add_memory(self, text: str, metadata: Dict[str, Any] | None = None) -> None:
        if not text.strip():
            return

        metadata = metadata or {}
        metadata.setdefault("source", "memory")
        metadata.setdefault("kind", "user_fact")
        
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        result = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        documents = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            out.append(
                {
                    "id": ids[i],
                    "text": doc,
                    "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                }
            )
        return out
