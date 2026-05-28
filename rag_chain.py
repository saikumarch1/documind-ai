from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("Loading components...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="llama3.2")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the document."
Always mention which page the answer came from.

Context: {context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join([f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain ready! Type 'quit' to exit\n")

while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    print("\nAnswer: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")