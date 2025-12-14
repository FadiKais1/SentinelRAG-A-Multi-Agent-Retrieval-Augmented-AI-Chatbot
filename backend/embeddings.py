from sentence_transformers import SentenceTransformer
from functools import lru_cache
from config import EMBED_MODEL

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def embed_texts(texts: list[str]):
    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True)  # <--- FORCE NUMPY
    return embeddings.tolist()  # <--- RETURN pure Python lists
