from backend.rag_pipeline import index_all_documents, answer_query

print("Indexing documents...")
count = index_all_documents()
print(f"Indexed {count} chunks.\n")

question = "What does RAG use for searching?"
print("Asking:", question)

result = answer_query(question)

print("\nANSWER:")
print(result["answer"])

print("\nCRITIQUE:")
print(result["critique"])

print("\nSOURCES:")
for ctx in result["context"]:
    print(f"- {ctx['metadata']['source']} (chunk {ctx['metadata']['chunk_index']})")
