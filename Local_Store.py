from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = Chroma(
    collection_name="strata_documents",
    embedding_function=embedder,
    persist_directory="./chroma_store",
)

print("Documents in store:", len(store.get()["ids"]))

query = "when was the invoice paid"
print("\nQuery:", query)

for doc, score in store.similarity_search_with_score(query, k=3):
    print(doc.metadata["doc_id"], round(score, 3), doc.page_content[:60])