from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

question = "What is a binary search tree?"
print(f"\nQuestion: {question}")

results = vectorstore.similarity_search_with_score(question, k=3)

for i, (doc, score) in enumerate(results):
    print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
    print(f"Page: {doc.metadata['page']}")
    print(f"Content: {doc.page_content[:200]}")