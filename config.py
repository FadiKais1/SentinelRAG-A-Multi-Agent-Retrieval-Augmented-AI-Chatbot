# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # ---------------------------------------
# # Load environment variables
# # ---------------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# load_dotenv(BASE_DIR / ".env")

# # ---------------------------------------
# # LLM Configuration
# # ---------------------------------------
# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")

# # Model name for NVIDIA / OpenAI
# LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mixtral-8x22b-instruct-v0.1")

# # NVIDIA
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# # OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# # ---------------------------------------
# # Embedding model
# # ---------------------------------------
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# # ---------------------------------------
# # Chunking configuration
# # ---------------------------------------
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# # ---------------------------------------
# # Vector DB directory
# # ---------------------------------------
# VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
# VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)



import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------
# Base paths & env
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------
# LLM configuration
# ---------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")

LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mixtral-8x22b-instruct-v0.1")

# NVIDIA
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ---------------------------------------
# Embedding model
# ---------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------
# Chunking configuration
# ---------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# ---------------------------------------
# Vector DB directory
# ---------------------------------------
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
