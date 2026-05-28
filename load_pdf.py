from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(pages)
    return chunks

# Test it
chunks = load_and_split("test.pdf")
print(f"Total chunks: {len(chunks)}")
print(f"\nChunk 1:\n{chunks[0].page_content}")
print(f"\nSource: page {chunks[0].metadata['page']}")