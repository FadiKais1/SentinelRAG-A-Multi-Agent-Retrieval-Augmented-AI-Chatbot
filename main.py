import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Embedding model (local, no API)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector database directory
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Chunk settings (used later)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
