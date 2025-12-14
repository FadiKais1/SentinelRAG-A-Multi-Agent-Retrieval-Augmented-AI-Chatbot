from backend.vector_store import VectorStore

store = VectorStore()

# Add one test document
docs = [
    {
        "id": "example_0",
        "text": "Artificial Intelligence systems like RAG use vector search.",
        "metadata": {"source": "example"}
    }
]

store.add_documents(docs)

print("Searching for: 'RAG'")
results = store.search("RAG", k=3)

print("\nResults:")
for r in results:
    print(f"- ID: {r['id']}")
    print(f"  Text: {r['text']}")
    print(f"  Metadata: {r['metadata']}")
    print()
