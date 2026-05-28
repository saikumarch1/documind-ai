from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Step 1: Loading PDF...")
loader = PyPDFLoader("test.pdf")
pages = loader.load()

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

print("Step 3: Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Step 4: Building FAISS index (this takes 2-3 minutes)...")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Step 5: Saving index to disk...")
vectorstore.save_local("faiss_index")

print("Done! Index saved.")