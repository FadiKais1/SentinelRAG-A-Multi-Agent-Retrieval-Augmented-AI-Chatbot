# from pathlib import Path
# from typing import List, Dict
# from config import BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# DOCS_DIR = BASE_DIR / "data" / "docs"

# def chunk_text(text: str) -> List[str]:
#     """
#     Splits text into overlapping word chunks.
#     """
#     words = text.split()
#     chunks = []
#     start = 0

#     while start < len(words):
#         end = min(start + CHUNK_SIZE, len(words))
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)

#         if end >= len(words):
#             break

#         start = end - CHUNK_OVERLAP

#     return chunks


# def load_documents() -> List[Dict]:
#     """
#     Loads ALL .txt files in /data/docs and splits them into chunks.
#     Returns a list of dicts for Chroma: id, text, metadata.
#     """
#     docs: List[Dict] = []

#     for file_path in DOCS_DIR.glob("*.txt"):
#         try:
#             text = file_path.read_text(encoding="utf-8")
#         except:
#             # fallback encoding
#             text = file_path.read_text(errors="ignore")

#         chunks = chunk_text(text)

#         for idx, chunk in enumerate(chunks):
#             docs.append({
#                 "id": f"{file_path.stem}_{idx}",
#                 "text": chunk,
#                 "metadata": {
#                     "source": file_path.name,
#                     "chunk_index": idx,
#                 }
#             })

#     return docs

#---------------------------------------------------------------------------------


from pathlib import Path
from typing import List, Dict
from config import BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

DOCS_DIR = BASE_DIR / "data" / "docs"


def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def _load_txt(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    print(f"[WARNING] Could not decode TXT file: {path.name}")
    return ""


def _load_pdf(path: Path) -> str:
    """
    Safe PDF loader.
    - Does NOT raise errors
    - Returns empty text if PDF unreadable
    """
    try:
        import PyPDF2
    except ImportError:
        print("[ERROR] PyPDF2 not installed.")
        return ""

    text = ""

    try:
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Some PDFs have no extractable text
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    continue

    except Exception as e:
        print(f"[WARNING] Could not read PDF '{path.name}': {e}")
        return ""

    return text



# def load_documents() -> List[Dict]:
#     """
#     Loads all .txt and .pdf files in data/docs, chunks them,
#     and returns a list of dicts ready for the vector store.
#     """
#     DOCS_DIR.mkdir(parents=True, exist_ok=True)

#     docs: List[Dict] = []

#     for file_path in DOCS_DIR.iterdir():
#         if file_path.suffix.lower() == ".txt":
#             raw_text = _load_txt(file_path)
#         elif file_path.suffix.lower() == ".pdf":
#             raw_text = _load_pdf(file_path)
#             if not raw_text.strip():
#                 print(f"[WARNING] Empty or unreadable PDF: {file_path.name}")
#                 continue

#         else:
#             continue

#         chunks = _chunk_text(raw_text)

#         for idx, chunk in enumerate(chunks):
#             docs.append(
#                 {
#                     "id": f"{file_path.name}_{idx}",
#                     "text": chunk,
#                     "metadata": {
#                         "source": file_path.name,
#                         "chunk_index": idx,
#                     },
#                 }
#             )

#     return docs



def load_documents() -> list[dict]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    docs = []

    for file_path in DOCS_DIR.iterdir():
        print("[DEBUG] Found file:", file_path.name, "suffix:", file_path.suffix)

        # --- Load file text ---
        if file_path.suffix.lower() == ".txt":
            raw_text = _load_txt(file_path)

        elif file_path.suffix.lower() == ".pdf":
            raw_text = _load_pdf(file_path)

        else:
            print("[DEBUG] Skipping unsupported file:", file_path.name)
            continue

        # ✅ ADD THE DEBUG LINE HERE
        print(f"[DEBUG] Loaded {file_path.name}, text length =", len(raw_text))

        # Skip empty files
        if not raw_text.strip():
            print(f"[WARNING] Empty or unreadable file: {file_path.name}")
            continue

        # --- Chunking ---
        chunks = _chunk_text(raw_text)
        print(f"[DEBUG] {file_path.name}: {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{file_path.name}_{idx}",  # IMPORTANT: unique ID
                    "text": chunk,
                    "metadata": {
                        "source": file_path.name,
                        "chunk_index": idx,
                    },
                }
            )

    return docs
